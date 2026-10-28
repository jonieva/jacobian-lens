# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Per-position residual extraction, Jacobian transport, unembedding, and concept rank readout.

Provides:
- :class:`PositionReadout`: Container holding token position, context token, per-layer raw/transported residuals, and logits.
- :func:`extract_position_readout`: Runs model and lens on a prompt at a position and computes per-layer residuals/logits.
- :func:`extract_residuals_at_position`: Fast batched unembedding returning residuals and top-K decoded words.
- :func:`encode_target_word`: Encodes a string into a single token ID (with warning if split across tokens).
- :func:`compute_word_ranks_from_readout`: Computes rank and Reciprocal Rank (RR) of a target token across layers.
- :func:`get_concept_rr_across_layers`: End-to-end convenience probe for a target word across layers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.vis import meaningful_token_mask

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Readout Data Structures & Extraction
# --------------------------------------------------------------------------- #


@dataclass
class PositionReadout:
    """Extracted residuals and decoded logits at a single token position across layers."""

    prompt: str
    input_ids: torch.Tensor
    position: int
    context_token_id: int
    context_token_str: str
    layers: list[int]
    raw_residuals: dict[int, torch.Tensor]
    transported_residuals: dict[int, torch.Tensor]
    logits: dict[int, torch.Tensor]
    final_raw_residual: torch.Tensor
    final_logits: torch.Tensor


@torch.no_grad()
def extract_position_readout(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    position: int = -1,
    layers: Sequence[int] | None = None,
    mask_display: bool = False,
    use_jacobian: bool = True,
    max_seq_len: int = 4000,
) -> PositionReadout:
    """Run ``model`` and ``lens`` on ``prompt`` at ``position`` and extract residuals and logits.

    Args:
        model: Target language model wrapped with :class:`~jlens.protocol.LensModel`.
        lens: Fitted :class:`~jlens.lens.JacobianLens`.
        prompt: Full text prompt to process.
        position: Token index to evaluate (supports negative indices, defaults to ``-1``).
        layers: Sequence of layer indices to extract. Defaults to all layers in ``lens``.
        mask_display: If True, masks unprintable / control token logits with ``-inf``.
        use_jacobian: If True, applies Jacobian transport; otherwise uses raw residual.
        max_seq_len: Truncation cutoff for input prompt.

    Returns:
        A populated :class:`PositionReadout`.
    """
    if layers is None:
        layers_list = sorted(lens.jacobians.keys())
    else:
        layers_list = list(layers)

    final_layer = model.n_layers - 1
    record_at = sorted(set(layers_list) | {final_layer})

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    abs_pos = position if position >= 0 else seq_len + position
    if not 0 <= abs_pos < seq_len:
        raise IndexError(f"position {position} (abs={abs_pos}) out of range for seq_len={seq_len}")

    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in record_at}

    raw_residuals: dict[int, torch.Tensor] = {}
    transported_residuals: dict[int, torch.Tensor] = {}
    logits: dict[int, torch.Tensor] = {}
    display_mask: torch.Tensor | None = None

    for layer in layers_list:
        h = activations[layer][0, abs_pos].float()
        raw_residuals[layer] = h.cpu()
        if use_jacobian and layer in lens.jacobians:
            h_trans = lens.transport(h.unsqueeze(0), layer)[0]
        else:
            h_trans = h
        transported_residuals[layer] = h_trans.cpu()

        layer_logits = model.unembed(h_trans.unsqueeze(0))[0].float().cpu()
        if mask_display:
            if display_mask is None:
                display_mask = meaningful_token_mask(
                    model.tokenizer, layer_logits.shape[-1], layer_logits.device
                )
            layer_logits = layer_logits.masked_fill(~display_mask, float("-inf"))
        logits[layer] = layer_logits

    h_final = activations[final_layer][0, abs_pos].float()
    final_logits = model.unembed(h_final.unsqueeze(0))[0].float().cpu()
    if mask_display:
        if display_mask is None:
            display_mask = meaningful_token_mask(
                model.tokenizer, final_logits.shape[-1], final_logits.device
            )
        final_logits = final_logits.masked_fill(~display_mask, float("-inf"))

    pos_tok_id = int(input_ids[0, abs_pos].item())
    pos_tok_str = model.tokenizer.decode(
        [pos_tok_id], clean_up_tokenization_spaces=False
    )

    return PositionReadout(
        prompt=prompt,
        input_ids=input_ids,
        position=abs_pos,
        context_token_id=pos_tok_id,
        context_token_str=pos_tok_str,
        layers=layers_list,
        raw_residuals=raw_residuals,
        transported_residuals=transported_residuals,
        logits=logits,
        final_raw_residual=h_final.cpu(),
        final_logits=final_logits,
    )


@torch.no_grad()
def extract_residuals_at_position(
    model: LensModel,
    lens: JacobianLens,
    full_prompt: str,
    position: int,
    layers: Sequence[int],
    *,
    use_jacobian: bool = True,
    max_seq_len: int = 4000,
    top_k: int = 10,
    mask_display: bool = True,
) -> tuple[
    int,
    int,
    str,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, list[int]],
    dict[int, list[str]],
    dict[int, list[int]],
    dict[int, list[str]],
]:
    """Extract raw/transported residuals and top-K decoded words at ``position`` across ``layers`` (batch unembedded).

    Returns:
        tuple containing:
        - absolute token position (int)
        - context token ID (int)
        - context token string (str)
        - raw residuals per layer {layer: tensor on CPU}
        - transported residuals per layer {layer: tensor on CPU}
        - top-K token IDs for transported residuals {layer: list[int]}
        - top-K token strings for transported residuals {layer: list[str]}
        - top-K token IDs for raw residuals {layer: list[int]}
        - top-K token strings for raw residuals {layer: list[str]}
    """
    layers_list = sorted(set(int(l) for l in layers))
    input_ids = model.encode(full_prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    abs_pos = position if position >= 0 else seq_len + position
    if not 0 <= abs_pos < seq_len:
        raise IndexError(
            f"position {position} (abs={abs_pos}) out of range for seq_len={seq_len}"
        )

    with ActivationRecorder(model.layers, at=layers_list) as recorder:
        model.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in layers_list}

    raw_residuals: dict[int, torch.Tensor] = {}
    transported_residuals: dict[int, torch.Tensor] = {}
    raw_on_device: list[torch.Tensor] = []
    trans_on_device: list[torch.Tensor] = []

    for layer in layers_list:
        h = activations[layer][0, abs_pos].float()
        raw_residuals[layer] = h.cpu()
        raw_on_device.append(h)
        if use_jacobian and layer in lens.jacobians:
            h_trans = lens.transport(h.unsqueeze(0), layer)[0]
        else:
            h_trans = h
        transported_residuals[layer] = h_trans.cpu()
        trans_on_device.append(h_trans)

    stacked_raw = torch.stack(raw_on_device, dim=0)
    stacked_trans = torch.stack(trans_on_device, dim=0)
    raw_logits = model.unembed(stacked_raw).float()
    trans_logits = model.unembed(stacked_trans).float()

    if mask_display:
        vocab_size = int(raw_logits.shape[-1])
        display_mask = meaningful_token_mask(
            model.tokenizer, vocab_size, raw_logits.device
        )
        raw_logits = raw_logits.masked_fill(~display_mask, float("-inf"))
        trans_logits = trans_logits.masked_fill(~display_mask, float("-inf"))

    top_trans_indices: list[list[int]] = (
        trans_logits.topk(top_k, dim=-1).indices.cpu().tolist()
    )
    top_raw_indices: list[list[int]] = (
        raw_logits.topk(top_k, dim=-1).indices.cpu().tolist()
    )

    unique_tids = {
        tid
        for row in (*top_trans_indices, *top_raw_indices)
        for tid in row
    }
    decoded_cache = {
        tid: model.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        for tid in unique_tids
    }

    top_k_ids: dict[int, list[int]] = {}
    top_k_tokens: dict[int, list[str]] = {}
    raw_top_k_ids: dict[int, list[int]] = {}
    raw_top_k_tokens: dict[int, list[str]] = {}

    for idx, layer in enumerate(layers_list):
        t_ids = top_trans_indices[idx]
        r_ids = top_raw_indices[idx]
        top_k_ids[layer] = t_ids
        top_k_tokens[layer] = [decoded_cache[tid] for tid in t_ids]
        raw_top_k_ids[layer] = r_ids
        raw_top_k_tokens[layer] = [decoded_cache[tid] for tid in r_ids]

    tok_id = int(input_ids[0, abs_pos].item())
    tok_str = model.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
    return (
        abs_pos,
        tok_id,
        tok_str,
        raw_residuals,
        transported_residuals,
        top_k_ids,
        top_k_tokens,
        raw_top_k_ids,
        raw_top_k_tokens,
    )


# --------------------------------------------------------------------------- #
# Concept Ranking & Vocabulary Probing Helpers
# --------------------------------------------------------------------------- #


def encode_target_word(
    tokenizer: Any,
    target_word: str,
    *,
    prepend_space: bool = True,
    verbose: bool = True,
) -> int:
    """Encode ``target_word`` to a single vocabulary token ID (warns if multi-token)."""
    word_to_encode = (" " + target_word.lstrip()) if prepend_space else target_word
    target_ids: list[int] = tokenizer.encode(word_to_encode, add_special_tokens=False)
    if len(target_ids) > 1 and verbose:
        decoded_parts = [tokenizer.decode([tid]) for tid in target_ids]
        logger.warning(
            "Target word %r tokenized into %d tokens %s (%s); tracking first token %d (%r).",
            word_to_encode,
            len(target_ids),
            target_ids,
            decoded_parts,
            target_ids[0],
            decoded_parts[0],
        )
    return target_ids[0]


def compute_word_ranks_from_readout(
    readout: PositionReadout,
    target_id: int,
) -> tuple[dict[int, float], dict[int, int], int]:
    """Compute ``(reciprocal_ranks, ranks, final_rank)`` for ``target_id`` from a :class:`PositionReadout`."""
    ranks: dict[int, int] = {}
    reciprocal_ranks: dict[int, float] = {}
    for layer in readout.layers:
        pos_logits = readout.logits[layer]
        target_score = pos_logits[target_id]
        rank = int((pos_logits > target_score).sum().item()) + 1
        ranks[layer] = rank
        reciprocal_ranks[layer] = 1.0 / rank

    final_score = readout.final_logits[target_id]
    final_rank = int((readout.final_logits > final_score).sum().item()) + 1
    return reciprocal_ranks, ranks, final_rank


def get_concept_rr_across_layers(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    target_word: str,
    layers: Sequence[int] | None = None,
    position: int = -1,
    prepend_space: bool = True,
    mask_display: bool = False,
) -> tuple[dict[int, float], dict[int, int], int, torch.Tensor]:
    """Compute 1-indexed rank and Reciprocal Rank (RR) of a target concept across layers.

    Returns:
        tuple containing:
        - reciprocal_ranks: {layer: float}
        - ranks: {layer: int}
        - final_rank: int
        - input_ids: torch.Tensor
    """
    target_id = encode_target_word(
        model.tokenizer, target_word, prepend_space=prepend_space
    )
    readout = extract_position_readout(
        model,
        lens,
        prompt,
        position=position,
        layers=layers,
        mask_display=mask_display,
    )
    seq_len = readout.input_ids.shape[1]
    target_tok_str = model.tokenizer.decode(
        [target_id], clean_up_tokenization_spaces=False
    )
    logger.info(
        "Tracking token_id=%d (%r) at pos=%d/%d (context token=%r)",
        target_id,
        target_tok_str,
        readout.position,
        seq_len - 1,
        readout.context_token_str,
    )
    reciprocal_ranks, ranks, final_rank = compute_word_ranks_from_readout(
        readout, target_id
    )
    return reciprocal_ranks, ranks, final_rank, readout.input_ids

