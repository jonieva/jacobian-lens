# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens loading, cross-lens cosine similarity, and cross-lingual representation comparison.

This module provides:
- Lens loading from HuggingFace Hub or local files (:func:`load_lens`, :func:`resolve_hub_lens_filename`)
- Layer-wise cosine similarity between two Jacobian lenses (:func:`get_lens_cosine_similarity`)
- Cross-lingual example comparisons across layers (:class:`CrossLingualComparison`, :func:`compare_examples_across_layers`, :func:`get_distance`)

For convenience and backward compatibility, this module also re-exports symbols from
:mod:`jlens.generation`, :mod:`jlens.readout`, and :mod:`jlens.experiments`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jlens.examples import Example, load_examples_from_yaml
from jlens.experiments import (
    EXPERIMENT_CSV_COLUMNS,
    GeneratedPromptReadout,
    compare_readouts_across_layers,
    compute_mean_similarities,
    generate_and_extract_readout,
    generate_and_extract_readouts,
    pair_examples_by_slug,
    parse_choice_from_response,
    recompute_experiment_from_csv,
    resolve_default_layers,
    run_comparison_experiment,
)
from jlens.generation import (
    ThinkingTemplateSpec,
    default_device,
    find_word_position_in_example,
    get_model_response,
    get_model_responses,
    get_thinking_template_spec,
    get_token_position_after_think_block,
    resolve_prompt_extended,
    set_seed_n,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import (
    PositionReadout,
    compute_word_ranks_from_readout,
    encode_target_word,
    extract_position_readout,
    extract_residuals_at_position,
    get_concept_rr_across_layers,
)

logger = logging.getLogger(__name__)

HUB_REPO_ID = "neuronpedia/jacobian-lens"

DEFAULT_HUB_LENS_FILES: dict[str, str] = {
    # Llama 3 models
    "meta-llama/Meta-Llama-3-8B": "llama-3-8b/jlens/Salesforce-wikitext/llama-3-8b_jacobian_lens.pt",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-it/jlens/Salesforce-wikitext/llama-3-8b-it_jacobian_lens.pt",
    "meta-llama/Meta-Llama-3-70B": "llama-3-70b/jlens/Salesforce-wikitext/llama-3-70b_jacobian_lens.pt",
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b-it/jlens/Salesforce-wikitext/llama-3-70b-it_jacobian_lens.pt",
    # Llama 3.1 models
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b/jlens/Salesforce-wikitext/llama-3.1-8b_jacobian_lens.pt",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-it/jlens/Salesforce-wikitext/llama-3.1-8b-it_jacobian_lens.pt",
    "meta-llama/Llama-3.1-70B": "llama-3.1-70b/jlens/Salesforce-wikitext/llama-3.1-70b_jacobian_lens.pt",
    "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b-it/jlens/Salesforce-wikitext/llama-3.1-70b-it_jacobian_lens.pt",
    # Qwen 2.5 models
    "Qwen/Qwen2.5-0.5B": "qwen-2.5-0.5b/jlens/Salesforce-wikitext/qwen-2.5-0.5b_jacobian_lens.pt",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-2.5-0.5b-it/jlens/Salesforce-wikitext/qwen-2.5-0.5b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-1.5B": "qwen-2.5-1.5b/jlens/Salesforce-wikitext/qwen-2.5-1.5b_jacobian_lens.pt",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-2.5-1.5b-it/jlens/Salesforce-wikitext/qwen-2.5-1.5b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-3B": "qwen-2.5-3b/jlens/Salesforce-wikitext/qwen-2.5-3b_jacobian_lens.pt",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-2.5-3b-it/jlens/Salesforce-wikitext/qwen-2.5-3b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-7B": "qwen-2.5-7b/jlens/Salesforce-wikitext/qwen-2.5-7b_jacobian_lens.pt",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b-it/jlens/Salesforce-wikitext/qwen-2.5-7b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-14B": "qwen-2.5-14b/jlens/Salesforce-wikitext/qwen-2.5-14b_jacobian_lens.pt",
    "Qwen/Qwen2.5-14B-Instruct": "qwen-2.5-14b-it/jlens/Salesforce-wikitext/qwen-2.5-14b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-32B": "qwen-2.5-32b/jlens/Salesforce-wikitext/qwen-2.5-32b_jacobian_lens.pt",
    "Qwen/Qwen2.5-32B-Instruct": "qwen-2.5-32b-it/jlens/Salesforce-wikitext/qwen-2.5-32b-it_jacobian_lens.pt",
    "Qwen/Qwen2.5-72B": "qwen-2.5-72b/jlens/Salesforce-wikitext/qwen-2.5-72b_jacobian_lens.pt",
    "Qwen/Qwen2.5-72B-Instruct": "qwen-2.5-72b-it/jlens/Salesforce-wikitext/qwen-2.5-72b-it_jacobian_lens.pt",
    # Qwen 3.5 & 3.6 models
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    # Gemma 2 models
    "google/gemma-2-2b": "gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt",
    "google/gemma-2-2b-it": "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt",
    "google/gemma-2-9b": "gemma-2-9b/jlens/Salesforce-wikitext/gemma-2-9b_jacobian_lens.pt",
    "google/gemma-2-9b-it": "gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt",
    "google/gemma-2-27b": "gemma-2-27b/jlens/Salesforce-wikitext/gemma-2-27b_jacobian_lens.pt",
    "google/gemma-2-27b-it": "gemma-2-27b-it/jlens/Salesforce-wikitext/gemma-2-27b-it_jacobian_lens.pt",
    # Gemma 3 & Gemma 4 models
    "google/gemma-3-1b-it": "gemma-3-1b-it/jlens/Salesforce-wikitext/gemma-3-1b-it_jacobian_lens.pt",
    "google/gemma-3-1b-pt": "gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt",
    "google/gemma-3-4b-it": "gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt",
    "google/gemma-3-4b-pt": "gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt",
    "google/gemma-3-12b-it": "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
    "google/gemma-3-12b-pt": "gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt",
    "google/gemma-3-27b-it": "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt",
    "google/gemma-3-27b-pt": "gemma-3-27b/jlens/Salesforce-wikitext/gemma-3-27b-pt_jacobian_lens.pt",
    "google/gemma-3-27b": "gemma-3-27b/jlens/Salesforce-wikitext/gemma-3-27b_jacobian_lens.pt",
    "google/gemma-4-E2B-it": "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt",
    "google/gemma-4-E2B": "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt",
    "google/gemma-4-E4B-it": "gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt",
    "google/gemma-4-E4B": "gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt",
    "google/gemma-4-31B-it": "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt",
    "google/gemma-4-31B": "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt",
}


def resolve_hub_lens_filename(model_name: str | None) -> str | None:
    """Resolve the default filename in neuronpedia/jacobian-lens for a given model."""
    if not model_name:
        return None
    if model_name in DEFAULT_HUB_LENS_FILES:
        return DEFAULT_HUB_LENS_FILES[model_name]
    # Check exact/substring match from most specific to least specific
    sorted_keys = sorted(DEFAULT_HUB_LENS_FILES.keys(), key=len, reverse=True)
    for key in sorted_keys:
        base = key.split("/")[-1].lower()
        if base in model_name.lower():
            return DEFAULT_HUB_LENS_FILES[key]
    return None


def load_lens(
    lens: str | None = None,
    lens_filename: str | None = None,
    lens_revision: str | None = None,
    model_name: str | None = None,
) -> JacobianLens:
    """Load a JacobianLens from either a custom local file or HuggingFace Hub.

    If a custom local file is provided (via ``lens`` or ``lens_filename``),
    it is loaded directly. If loading from a Hub repo (such as ``neuronpedia/jacobian-lens``)
    and ``lens_filename`` is omitted, the path inside the repository is automatically
    inferred from ``model_name``.
    """
    # 1. Local file passed via lens_filename
    if lens_filename and (os.path.isfile(lens_filename) or not lens or lens_filename.endswith(".pt")):
        if not os.path.isfile(lens_filename):
            p = Path(lens_filename)
            if p.parent.exists():
                candidates = list(p.parent.glob("*.pt"))
                if candidates:
                    candidates_str = ", ".join(f"'{c}'" for c in candidates)
                    raise FileNotFoundError(
                        f"Custom lens file not found: {lens_filename!r}. "
                        f"Available .pt files in {p.parent}: {candidates_str}"
                    )
            raise FileNotFoundError(f"Custom lens file not found: {lens_filename!r}")
        logger.info("Loading custom Jacobian lens from file: %r", lens_filename)
        return JacobianLens.load(lens_filename)

    # 2. Local file passed via lens
    if lens and (os.path.isfile(lens) or lens.endswith(".pt")):
        if not os.path.isfile(lens):
            raise FileNotFoundError(f"Custom lens file not found: {lens!r}")
        logger.info("Loading custom Jacobian lens from file: %r", lens)
        return JacobianLens.load(lens)

    # 3. HuggingFace Hub repo
    if lens:
        filename = lens_filename or resolve_hub_lens_filename(model_name)
        if not filename:
            if "neuronpedia" in lens:
                known_models = "\n  - " + "\n  - ".join(sorted(DEFAULT_HUB_LENS_FILES.keys()))
                raise ValueError(
                    f"Could not automatically resolve default lens filename for model {model_name!r} "
                    f"in HuggingFace repo {lens!r}.\n"
                    f"Please specify --lens-filename explicitly (e.g. '<model_dir>/jlens/Salesforce-wikitext/<model>_jacobian_lens.pt').\n"
                    f"Known models in {lens!r}:{known_models}"
                )
            filename = "lens.pt"

        revision = lens_revision
        if revision is None and "neuronpedia" in lens:
            revision = "qwen-n1000" if (model_name and "qwen" in model_name.lower()) else "main"

        logger.info(
            "Loading Jacobian lens from HuggingFace repo %r (filename=%r, revision=%r)...",
            lens,
            filename,
            revision,
        )
        return JacobianLens.from_pretrained(
            lens,
            filename=filename,
            revision=revision,
        )

    raise ValueError(
        "No lens specified. Please provide a custom lens file via lens or lens_filename, "
        "or a HuggingFace Hub repo via lens."
    )


def get_lens_cosine_similarity(lens_1: JacobianLens, lens_2: JacobianLens) -> list[float]:
    """Compute cosine similarity between the flattened transport matrices of two lenses across layers."""
    cosine_sims: list[float] = []
    common_layers = sorted(set(lens_1.jacobians.keys()) & set(lens_2.jacobians.keys()))
    for layer_idx in common_layers:
        j1 = lens_1.jacobians[layer_idx].float()
        j2 = lens_2.jacobians[layer_idx].float()
        cos_sim = float(F.cosine_similarity(j1.flatten(), j2.flatten(), dim=0).item())
        cosine_sims.append(cos_sim)
    return cosine_sims


# --------------------------------------------------------------------------- #
# Cross-Lingual Comparisons
# --------------------------------------------------------------------------- #


@dataclass
class CrossLingualComparison:
    """Layer-by-layer comparison between an English and a Spanish :class:`~jlens.examples.Example`."""

    layers: list[int]
    pos_en: int
    pos_es: int
    readout_en: PositionReadout
    readout_es: PositionReadout
    raw_cosine_distance: dict[int, float]
    raw_cosine_similarity: dict[int, float]
    transported_cosine_distance: dict[int, float]
    transported_cosine_similarity: dict[int, float]
    recall_at_k: dict[int, float]
    shared_top_k_tokens: dict[int, list[str]]
    ranks_en: dict[int, int] = field(default_factory=dict)
    rr_en: dict[int, float] = field(default_factory=dict)
    hit_at_k_en: dict[int, bool] = field(default_factory=dict)
    ranks_es: dict[int, int] = field(default_factory=dict)
    rr_es: dict[int, float] = field(default_factory=dict)
    hit_at_k_es: dict[int, bool] = field(default_factory=dict)


def compare_examples_across_layers(
    model: LensModel,
    lens: JacobianLens,
    example_en: Example,
    example_es: Example,
    pos_en: int | Sequence[int] | str,
    pos_es: int | Sequence[int] | str,
    layers: Sequence[int] | None = None,
    *,
    target_word: str | None = None,
    k: int = 10,
    mask_display: bool = True,
    prepend_space: bool = True,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
    verbose: bool = True,
) -> CrossLingualComparison:
    """Compare residual cosine distance, Recall@K, and concept RR between EN and ES examples."""
    tokenizer = model.tokenizer
    if isinstance(pos_en, str):
        resolved_pos_en = find_word_position_in_example(
            example_en,
            pos_en,
            tokenizer,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
        )[-1]
    elif isinstance(pos_en, int):
        resolved_pos_en = pos_en
    else:
        resolved_pos_en = int(pos_en[-1])

    if isinstance(pos_es, str):
        resolved_pos_es = find_word_position_in_example(
            example_es,
            pos_es,
            tokenizer,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
        )[-1]
    elif isinstance(pos_es, int):
        resolved_pos_es = pos_es
    else:
        resolved_pos_es = int(pos_es[-1])

    prompt_en = resolve_prompt_extended(
        example_en,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    prompt_es = resolve_prompt_extended(
        example_es,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )

    readout_en = extract_position_readout(
        model,
        lens,
        prompt_en,
        position=resolved_pos_en,
        layers=layers,
        mask_display=mask_display,
    )
    readout_es = extract_position_readout(
        model,
        lens,
        prompt_es,
        position=resolved_pos_es,
        layers=layers,
        mask_display=mask_display,
    )

    raw_cos_dist: dict[int, float] = {}
    raw_cos_sim: dict[int, float] = {}
    trans_cos_dist: dict[int, float] = {}
    trans_cos_sim: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    shared_top_k_tokens: dict[int, list[str]] = {}

    for layer in readout_en.layers:
        h_en = readout_en.raw_residuals[layer]
        h_es = readout_es.raw_residuals[layer]
        r_sim = float(F.cosine_similarity(h_en, h_es, dim=0).item())
        raw_cos_sim[layer] = r_sim
        raw_cos_dist[layer] = 1.0 - r_sim

        jh_en = readout_en.transported_residuals[layer]
        jh_es = readout_es.transported_residuals[layer]
        t_sim = float(F.cosine_similarity(jh_en, jh_es, dim=0).item())
        trans_cos_sim[layer] = t_sim
        trans_cos_dist[layer] = 1.0 - t_sim

        top_en_ids = readout_en.logits[layer].topk(k).indices.tolist()
        top_es_ids_set = set(readout_es.logits[layer].topk(k).indices.tolist())
        shared_ids = [tid for tid in top_en_ids if tid in top_es_ids_set]
        recall_at_k[layer] = len(shared_ids) / float(k)
        shared_top_k_tokens[layer] = [
            tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            for tid in shared_ids
        ]

    ranks_en: dict[int, int] = {}
    rr_en: dict[int, float] = {}
    hit_at_k_en: dict[int, bool] = {}
    ranks_es: dict[int, int] = {}
    rr_es: dict[int, float] = {}
    hit_at_k_es: dict[int, bool] = {}

    if target_word is not None:
        target_id = encode_target_word(
            tokenizer, target_word, prepend_space=prepend_space, verbose=verbose
        )
        rr_en, ranks_en, _ = compute_word_ranks_from_readout(readout_en, target_id)
        rr_es, ranks_es, _ = compute_word_ranks_from_readout(readout_es, target_id)
        hit_at_k_en = {layer: (ranks_en[layer] <= k) for layer in readout_en.layers}
        hit_at_k_es = {layer: (ranks_es[layer] <= k) for layer in readout_es.layers}

    return CrossLingualComparison(
        layers=readout_en.layers,
        pos_en=readout_en.position,
        pos_es=readout_es.position,
        readout_en=readout_en,
        readout_es=readout_es,
        raw_cosine_distance=raw_cos_dist,
        raw_cosine_similarity=raw_cos_sim,
        transported_cosine_distance=trans_cos_dist,
        transported_cosine_similarity=trans_cos_sim,
        recall_at_k=recall_at_k,
        shared_top_k_tokens=shared_top_k_tokens,
        ranks_en=ranks_en,
        rr_en=rr_en,
        hit_at_k_en=hit_at_k_en,
        ranks_es=ranks_es,
        rr_es=rr_es,
        hit_at_k_es=hit_at_k_es,
    )


def get_distance(
    model: LensModel,
    lens: JacobianLens,
    example_en: Example,
    example_es: Example,
    pos_en: int | Sequence[int] | str,
    pos_es: int | Sequence[int] | str,
    layers: Sequence[int] | None = None,
    *,
    mask_display: bool = True,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
) -> dict[int, float]:
    """Return transported cosine similarity between two examples at the specified positions."""
    tokenizer = model.tokenizer
    prompt_en = resolve_prompt_extended(
        example_en,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    prompt_es = resolve_prompt_extended(
        example_es,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    readout_en = extract_position_readout(
        model,
        lens,
        prompt_en,
        position=pos_en if isinstance(pos_en, int) else int(pos_en[-1]),
        layers=layers,
        mask_display=mask_display,
    )
    readout_es = extract_position_readout(
        model,
        lens,
        prompt_es,
        position=pos_es if isinstance(pos_es, int) else int(pos_es[-1]),
        layers=layers,
        mask_display=mask_display,
    )
    trans_cos_sim: dict[int, float] = {}
    for layer in readout_en.layers:
        jh_en = readout_en.transported_residuals[layer]
        jh_es = readout_es.transported_residuals[layer]
        trans_cos_sim[layer] = float(F.cosine_similarity(jh_en, jh_es, dim=0).item())
    return trans_cos_sim


__all__ = [
    # Hub & Lens Loading
    "DEFAULT_HUB_LENS_FILES",
    "HUB_REPO_ID",
    "resolve_hub_lens_filename",
    "load_lens",
    "get_lens_cosine_similarity",
    # Representation Comparisons
    "CrossLingualComparison",
    "compare_examples_across_layers",
    "get_distance",
    # Re-exported from jlens.generation
    "default_device",
    "set_seed_n",
    "ThinkingTemplateSpec",
    "get_thinking_template_spec",
    "resolve_prompt_extended",
    "get_token_position_after_think_block",
    "find_word_position_in_example",
    "get_model_response",
    "get_model_responses",
    # Re-exported from jlens.readout
    "PositionReadout",
    "extract_position_readout",
    "extract_residuals_at_position",
    "encode_target_word",
    "compute_word_ranks_from_readout",
    "get_concept_rr_across_layers",
    # Re-exported from jlens.experiments
    "GeneratedPromptReadout",
    "resolve_default_layers",
    "pair_examples_by_slug",
    "generate_and_extract_readout",
    "generate_and_extract_readouts",
    "compare_readouts_across_layers",
    "parse_choice_from_response",
    "EXPERIMENT_CSV_COLUMNS",
    "run_comparison_experiment",
    "recompute_experiment_from_csv",
    "compute_mean_similarities",
    # Re-exported from jlens.examples
    "load_examples_from_yaml",
]
