# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Cross-lingual evaluation pipeline, batch comparison runners, and CSV persistence/analysis.

Provides:
- :class:`GeneratedPromptReadout`: Container storing prompt generation, post-think token, and per-layer residuals/top-K tokens.
- :func:`resolve_default_layers`: Selects layers to evaluate (strided or explicit), always ending at the final layer.
- :func:`pair_examples_by_slug`: Aligns examples across two languages matching on ``slug``.
- :func:`generate_and_extract_readout`: Runs model generation, post-think token location, and residual extraction for an example.
- :func:`compare_readouts_across_layers`: Computes layer-wise cosine similarities and token overlaps between two readouts.
- :func:`run_comparison_experiment`: Full pipeline executing L1/L1, L2/L2, and L1/L2 comparisons with CSV streaming.
- :func:`recompute_experiment_from_csv`: Recomputes lens transport and similarities from previously logged CSV prompts/positions.
- :func:`compute_mean_similarities`: Aggregates mean/std cosine similarity and top-K overlap per scenario and language grouping.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jlens.examples import Example, load_examples_from_yaml
from jlens.generation import (
    get_model_response,
    get_model_responses,
    get_thinking_template_spec,
    get_token_position_after_think_block,
    resolve_prompt_extended,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import extract_residuals_at_position

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data Containers & Helpers
# --------------------------------------------------------------------------- #


@dataclass
class GeneratedPromptReadout:
    """Generated response, full prompt, post-think token position, per-layer residuals, and top-K tokens."""

    slug: str
    language: str
    base_prompt: str
    response: str
    full_prompt: str
    position: int
    context_token_id: int
    context_token_str: str
    layers: list[int]
    raw_residuals: dict[int, torch.Tensor]
    transported_residuals: dict[int, torch.Tensor]
    top_k_ids: dict[int, list[int]] = field(default_factory=dict)
    top_k_tokens: dict[int, list[str]] = field(default_factory=dict)
    raw_top_k_ids: dict[int, list[int]] = field(default_factory=dict)
    raw_top_k_tokens: dict[int, list[str]] = field(default_factory=dict)
    answer: int | None = None


def resolve_default_layers(
    n_layers: int,
    layers: Sequence[int] | None = None,
    *,
    stride: int = 2,
) -> list[int]:
    """Return sorted layer indices to evaluate, always including the final layer."""
    final_layer = n_layers - 1
    if layers is not None:
        resolved = sorted(set(int(l) for l in layers))
    else:
        actual_start = n_layers // 2    # Middle layer
        resolved = list(range(actual_start, final_layer, stride))
    if final_layer not in resolved:
        resolved.append(final_layer)
    return sorted(set(resolved))


def pair_examples_by_slug(
    examples_l1: Sequence[Example],
    examples_l2: Sequence[Example],
) -> list[tuple[Example, Example]]:
    """Pair examples from two lists that share an identical ``slug``."""
    l2_by_slug = {ex.slug: ex for ex in examples_l2}
    paired: list[tuple[Example, Example]] = []
    seen: set[str] = set()
    for ex1 in examples_l1:
        if ex1.slug in l2_by_slug and ex1.slug not in seen:
            paired.append((ex1, l2_by_slug[ex1.slug]))
            seen.add(ex1.slug)
    return paired


def generate_and_extract_readouts(
    hf_model: Any,
    model: LensModel,
    lens: JacobianLens,
    example: Example,
    language: str,
    layers: Sequence[int],
    *,
    n_samples: int = 1,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
    max_new_tokens: int = 3000,
    max_seq_len: int = 4000,
    max_retries: int = 3,
    top_k: int = 10,
    mask_display: bool = True,
) -> list[GeneratedPromptReadout]:
    """Resolve ``example``, generate ``n_samples`` responses in a single batched call, and extract per-layer residuals."""
    if n_samples <= 0:
        return []
    tokenizer = model.tokenizer
    base_prompt = resolve_prompt_extended(
        example,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )

    responses = get_model_responses(
        hf_model,
        tokenizer,
        base_prompt,
        n=n_samples,
        max_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )

    readouts: list[GeneratedPromptReadout] = []
    for response in responses:
        full_prompt = base_prompt + response
        pos = get_token_position_after_think_block(
            tokenizer, full_prompt, base_prompt=base_prompt, max_seq_len=max_seq_len
        )

        if pos == -1:
            raise RuntimeError(
                f"Failed to locate post-think / pre-answer token for slug={example.slug!r} ({language})."
            )

        (
            abs_pos,
            tok_id,
            tok_str,
            raw_res,
            trans_res,
            top_ids,
            top_toks,
            raw_top_ids,
            raw_top_toks,
        ) = extract_residuals_at_position(
            model,
            lens,
            full_prompt,
            position=pos,
            layers=layers,
            use_jacobian=True,
            max_seq_len=max_seq_len,
            top_k=top_k,
            mask_display=mask_display,
        )

        ans = parse_choice_from_response(response, tokenizer=tokenizer)
        readouts.append(
            GeneratedPromptReadout(
                slug=example.slug,
                language=language,
                base_prompt=base_prompt,
                response=response,
                full_prompt=full_prompt,
                position=abs_pos,
                context_token_id=tok_id,
                context_token_str=tok_str,
                layers=list(layers),
                raw_residuals=raw_res,
                transported_residuals=trans_res,
                top_k_ids=top_ids,
                top_k_tokens=top_toks,
                raw_top_k_ids=raw_top_ids,
                raw_top_k_tokens=raw_top_toks,
                answer=ans,
            )
        )

    return readouts


def generate_and_extract_readout(
    hf_model: Any,
    model: LensModel,
    lens: JacobianLens,
    example: Example,
    language: str,
    layers: Sequence[int],
    *,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
    max_new_tokens: int = 3000,
    max_seq_len: int = 4000,
    max_retries: int = 3,
    top_k: int = 10,
    mask_display: bool = True,
) -> GeneratedPromptReadout:
    """Resolve ``example``, generate a single response, and extract per-layer residuals."""
    return generate_and_extract_readouts(
        hf_model,
        model,
        lens,
        example,
        language,
        layers,
        n_samples=1,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        max_retries=max_retries,
        top_k=top_k,
        mask_display=mask_display,
    )[0]


def parse_choice_from_response(
    response_text: str | None,
    tokenizer: Any = None,
) -> int | None:
    """Attempt to parse the chosen option index (e.g. 1 or 2) from a model response.

    If ``tokenizer`` is provided, delimiters and special tokens are dynamically
    inferred from ``get_thinking_template_spec(tokenizer)`` and ``tokenizer.all_special_tokens``.
    Otherwise, standard delimiter patterns are used as fallbacks.
    Returns None if no option index can be identified.
    """
    if not response_text:
        return None

    text = str(response_text)

    # 1. Strip thinking block
    closing_tags: list[str] = []
    if tokenizer is not None:
        spec = get_thinking_template_spec(tokenizer)
        if spec.close_tag:
            closing_tags.append(spec.close_tag)
        if spec.close_suffix and spec.close_suffix.strip() not in closing_tags:
            closing_tags.append(spec.close_suffix.strip())

    # Fallback tags if tokenizer not provided or closing tag not found
    for fallback_tag in ("</think>", "<channel|>", "<|/thought|>", "</thought>"):
        if fallback_tag not in closing_tags:
            closing_tags.append(fallback_tag)

    for tag in closing_tags:
        if tag in text:
            text = text.split(tag)[-1]
            break

    # 2. Strip special tokens if tokenizer is available
    if tokenizer is not None and hasattr(tokenizer, "all_special_tokens"):
        for sp in tokenizer.all_special_tokens:
            if sp and sp in text:
                text = text.replace(sp, " ")

    # Remove remaining generic control/special token-like markers (e.g. <turn|>, <|im_end|>)
    text = re.sub(r"<\|?[a-zA-Z0-9_-]+\|?>", " ", text).strip()
    if not text:
        # Fallback: if output was placed right before the thinking close tag
        raw = str(response_text)
        for tag in closing_tags:
            if tag in raw:
                before = raw.split(tag)[0][-150:]
                m = re.findall(
                    r"(?:Output|Answer|Respuesta|Choice|Opción|Option|Acción|Action)?\s*[:#-]?\s*([1-9][0-9]?)",
                    before,
                    re.IGNORECASE,
                )
                if m:
                    for val in reversed(m):
                        if val.strip():
                            return int(val)
        return None

    # 3. Direct digit or prefixed digit at the start of answer: '1', ' 2 ', 'Option 1', 'Acción 2', etc.
    m = re.match(
        r"^(?:(?:Option|Action|Opción|Acción|Choice|Number|Respuesta|Número)\s*[:#-]?\s*)?([1-9][0-9]?)(?:[.:)\s]|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # 4. Decision phrases: 'choose 1', 'elijo 2', 'select 1', etc.
    m = re.search(
        r"(?:choose|pick|select|elijo|elija|opción|opcion|action|acción|accion)\s*[:#-]?\s*([1-9][0-9]?)",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # 5. First standalone digit if answer text is relatively short (< 100 characters)
    if len(text) < 100:
        m = re.search(r"\b([1-9][0-9]?)\b", text)
        if m:
            return int(m.group(1))

    return None


def compare_readouts_across_layers(
    readout_1: GeneratedPromptReadout,
    readout_2: GeneratedPromptReadout,
    *,
    scenario: str,
    sample_id: int = 0,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Create per-layer comparison rows for a pair of :class:`GeneratedPromptReadout`."""
    rows: list[dict[str, Any]] = []
    same_language = readout_1.language == readout_2.language
    ans_1 = (
        readout_1.answer
        if readout_1.answer is not None
        else parse_choice_from_response(readout_1.response, tokenizer=tokenizer)
    )
    ans_2 = (
        readout_2.answer
        if readout_2.answer is not None
        else parse_choice_from_response(readout_2.response, tokenizer=tokenizer)
    )
    same_answer = (ans_1 == ans_2) if (ans_1 is not None and ans_2 is not None) else None
    for layer in readout_1.layers:
        jh_1 = readout_1.transported_residuals[layer]
        jh_2 = readout_2.transported_residuals[layer]
        cos_sim = float(F.cosine_similarity(jh_1, jh_2, dim=0).item())

        h_1 = readout_1.raw_residuals[layer]
        h_2 = readout_2.raw_residuals[layer]
        raw_cos_sim = float(F.cosine_similarity(h_1, h_2, dim=0).item())

        top_ids_1 = readout_1.top_k_ids.get(layer, [])
        top_ids_2 = readout_2.top_k_ids.get(layer, [])
        top_toks_1 = readout_1.top_k_tokens.get(layer, [])
        top_toks_2 = readout_2.top_k_tokens.get(layer, [])
        top_ids_2_set = set(top_ids_2)
        shared_tokens = [
            tok for tid, tok in zip(top_ids_1, top_toks_1) if tid in top_ids_2_set
        ]
        overlap_count = len(shared_tokens)
        k_trans = max(len(top_ids_1), len(top_ids_2), 1)
        overlap_ratio = overlap_count / float(k_trans)

        raw_ids_1 = readout_1.raw_top_k_ids.get(layer, [])
        raw_ids_2 = readout_2.raw_top_k_ids.get(layer, [])
        raw_toks_1 = readout_1.raw_top_k_tokens.get(layer, [])
        raw_toks_2 = readout_2.raw_top_k_tokens.get(layer, [])
        raw_ids_2_set = set(raw_ids_2)
        raw_shared_tokens = [
            tok for tid, tok in zip(raw_ids_1, raw_toks_1) if tid in raw_ids_2_set
        ]
        raw_overlap_count = len(raw_shared_tokens)
        k_raw = max(len(raw_ids_1), len(raw_ids_2), 1)
        raw_overlap_ratio = raw_overlap_count / float(k_raw)

        rows.append(
            {
                "slug": readout_1.slug,
                "scenario": scenario,
                "lang_1": readout_1.language,
                "lang_2": readout_2.language,
                "same_language": same_language,
                "sample_id": sample_id,
                "layer": layer,
                "pos_1": readout_1.position,
                "pos_2": readout_2.position,
                "token_1": readout_1.context_token_str,
                "token_2": readout_2.context_token_str,
                "response_1": readout_1.response,
                "response_2": readout_2.response,
                "answer_1": ans_1,
                "answer_2": ans_2,
                "same_answer": same_answer,
                "full_prompt_1": readout_1.full_prompt,
                "full_prompt_2": readout_2.full_prompt,
                "cosine_similarity": cos_sim,
                "raw_cosine_similarity": raw_cos_sim,
                "top_k_overlap_count": overlap_count,
                "top_k_overlap_ratio": overlap_ratio,
                "raw_top_k_overlap_count": raw_overlap_count,
                "raw_top_k_overlap_ratio": raw_overlap_ratio,
                "top_k_tokens_1": json.dumps(top_toks_1, ensure_ascii=False),
                "top_k_tokens_2": json.dumps(top_toks_2, ensure_ascii=False),
                "shared_top_k_tokens": json.dumps(shared_tokens, ensure_ascii=False),
                "raw_top_k_tokens_1": json.dumps(raw_toks_1, ensure_ascii=False),
                "raw_top_k_tokens_2": json.dumps(raw_toks_2, ensure_ascii=False),
                "raw_shared_top_k_tokens": json.dumps(
                    raw_shared_tokens, ensure_ascii=False
                ),
            }
        )
    return rows


EXPERIMENT_CSV_COLUMNS: list[str] = [
    "slug",
    "scenario",
    "lang_1",
    "lang_2",
    "same_language",
    "sample_id",
    "layer",
    "pos_1",
    "pos_2",
    "token_1",
    "token_2",
    "response_1",
    "response_2",
    "answer_1",
    "answer_2",
    "same_answer",
    "full_prompt_1",
    "full_prompt_2",
    "cosine_similarity",
    "raw_cosine_similarity",
    "top_k_overlap_count",
    "top_k_overlap_ratio",
    "raw_top_k_overlap_count",
    "raw_top_k_overlap_ratio",
    "top_k_tokens_1",
    "top_k_tokens_2",
    "shared_top_k_tokens",
    "raw_top_k_tokens_1",
    "raw_top_k_tokens_2",
    "raw_shared_top_k_tokens",
]


# --------------------------------------------------------------------------- #
# Experiment Execution & CSV Persistence
# --------------------------------------------------------------------------- #


def run_comparison_experiment(
    hf_model: Any,
    model: LensModel,
    lens: JacobianLens,
    en_yaml_path: str | Path,
    es_yaml_path: str | Path,
    output_csv_path: str | Path,
    *,
    layers: Sequence[int] | None = None,
    lang1_label: str = "English",
    lang2_label: str = "Spanish",
    n_bootstrap: int = 1,
    include_single_language: bool = True,
    enable_thinking: bool = True,
    reasoning_effort: str = "medium",
    max_new_tokens: int = 3000,
    max_seq_len: int = 4000,
    top_k: int = 10,
    mask_display: bool = True,
) -> pd.DataFrame:
    """Run cross-lingual post-think residual cosine similarity experiment and stream results to CSV."""
    examples_en = load_examples_from_yaml(en_yaml_path)
    examples_es = load_examples_from_yaml(es_yaml_path)
    paired_examples = pair_examples_by_slug(examples_en, examples_es)

    if not paired_examples:
        raise ValueError(
            f"No matching slugs found between {en_yaml_path} and {es_yaml_path}"
        )

    target_layers = resolve_default_layers(model.n_layers, layers)
    logger.info(
        "Found %d paired examples. Evaluating layers: %s (top_k=%d, mask_display=%s)",
        len(paired_examples),
        target_layers,
        top_k,
        mask_display,
    )

    all_rows: list[dict[str, Any]] = []
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=EXPERIMENT_CSV_COLUMNS).to_csv(output_path, index=False)

    scenario_l1_l1 = f"{lang1_label}/{lang1_label}"
    scenario_l2_l2 = f"{lang2_label}/{lang2_label}"
    scenario_l1_l2 = f"{lang1_label}/{lang2_label}"

    total_iterations = len(paired_examples) * n_bootstrap
    with tqdm(total=total_iterations, desc="Progress", unit="iter") as pbar:
        for ex_en, ex_es in paired_examples:
            for sample_id in range(n_bootstrap):
                pbar.set_postfix(
                    slug=ex_en.slug,
                    sample=f"{sample_id + 1}/{n_bootstrap}",
                    rows=len(all_rows),
                )
                if include_single_language:
                    readout_l1_a, readout_l1_b = generate_and_extract_readouts(
                        hf_model,
                        model,
                        lens,
                        ex_en,
                        lang1_label,
                        target_layers,
                        n_samples=2,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        top_k=top_k,
                        mask_display=mask_display,
                    )
                    all_rows.extend(
                        compare_readouts_across_layers(
                            readout_l1_a,
                            readout_l1_b,
                            scenario=scenario_l1_l1,
                            sample_id=sample_id,
                        )
                    )

                    readout_l2_a, readout_l2_b = generate_and_extract_readouts(
                        hf_model,
                        model,
                        lens,
                        ex_es,
                        lang2_label,
                        target_layers,
                        n_samples=2,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        top_k=top_k,
                        mask_display=mask_display,
                    )
                    all_rows.extend(
                        compare_readouts_across_layers(
                            readout_l1_a,
                            readout_l2_a,
                            scenario=scenario_l1_l2,
                            sample_id=sample_id,
                        )
                    )
                    all_rows.extend(
                        compare_readouts_across_layers(
                            readout_l2_a,
                            readout_l2_b,
                            scenario=scenario_l2_l2,
                            sample_id=sample_id,
                        )
                    )
                else:
                    readout_l1_a = generate_and_extract_readout(
                        hf_model,
                        model,
                        lens,
                        ex_en,
                        lang1_label,
                        target_layers,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        top_k=top_k,
                        mask_display=mask_display,
                    )
                    readout_l2_a = generate_and_extract_readout(
                        hf_model,
                        model,
                        lens,
                        ex_es,
                        lang2_label,
                        target_layers,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        top_k=top_k,
                        mask_display=mask_display,
                    )
                    all_rows.extend(
                        compare_readouts_across_layers(
                            readout_l1_a,
                            readout_l2_a,
                            scenario=scenario_l1_l2,
                            sample_id=sample_id,
                        )
                    )

                pd.DataFrame(all_rows).to_csv(output_path, index=False)
                pbar.update(1)

    df = pd.DataFrame(all_rows)
    logger.info("Finished. Total rows: %d in %s", len(df), output_path)
    return df


def recompute_experiment_from_csv(
    model: LensModel,
    lens: JacobianLens,
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    layers: Sequence[int] | None = None,
    max_seq_len: int = 4000,
    top_k: int = 10,
    mask_display: bool = True,
) -> pd.DataFrame:
    """Recompute cosine similarities and top-K words using prompts/positions saved in ``input_csv_path``."""
    src_df = pd.read_csv(input_csv_path)
    if src_df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv_path}")

    if layers is not None:
        target_layers = resolve_default_layers(model.n_layers, layers)
    elif "layer" in src_df.columns:
        target_layers = sorted(set(int(l) for l in src_df["layer"].unique()))
    else:
        target_layers = resolve_default_layers(model.n_layers, None)

    if "sample_id" not in src_df.columns:
        src_df["sample_id"] = 0

    groups = list(src_df.groupby(["slug", "sample_id", "scenario"], sort=False))
    logger.info(
        "Recomputing %d scenario comparisons from %s across layers %s...",
        len(groups),
        input_csv_path,
        target_layers,
    )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    readout_cache: dict[tuple[str, int, str, str, int], GeneratedPromptReadout] = {}

    def _get_readout(
        slug: str,
        sample_id: int,
        lang: str,
        response: str,
        full_prompt: str,
        pos: int,
    ) -> GeneratedPromptReadout:
        key = (slug, sample_id, lang, full_prompt, pos)
        if key not in readout_cache:
            (
                abs_pos,
                tok_id,
                tok_str,
                raw_res,
                trans_res,
                top_ids,
                top_toks,
                raw_top_ids,
                raw_top_toks,
            ) = extract_residuals_at_position(
                model,
                lens,
                full_prompt,
                position=pos,
                layers=target_layers,
                use_jacobian=True,
                max_seq_len=max_seq_len,
                top_k=top_k,
                mask_display=mask_display,
            )
            ans = parse_choice_from_response(str(response), tokenizer=model.tokenizer)
            readout_cache[key] = GeneratedPromptReadout(
                slug=slug,
                language=lang,
                base_prompt="",
                response=str(response),
                full_prompt=full_prompt,
                position=abs_pos,
                context_token_id=tok_id,
                context_token_str=tok_str,
                layers=list(target_layers),
                raw_residuals=raw_res,
                transported_residuals=trans_res,
                top_k_ids=top_ids,
                top_k_tokens=top_toks,
                raw_top_k_ids=raw_top_ids,
                raw_top_k_tokens=raw_top_toks,
                answer=ans,
            )
        return readout_cache[key]

    current_slug_sample: tuple[str, int] | None = None
    with tqdm(total=len(groups), desc="Recomputing", unit="pair") as pbar:
        for (slug, sample_id, scenario), grp in groups:
            if current_slug_sample != (str(slug), int(sample_id)):
                readout_cache.clear()
                current_slug_sample = (str(slug), int(sample_id))

            first_row = grp.iloc[0]
            readout_1 = _get_readout(
                slug=str(slug),
                sample_id=int(sample_id),
                lang=str(first_row["lang_1"]),
                response=str(first_row["response_1"]),
                full_prompt=str(first_row["full_prompt_1"]),
                pos=int(first_row["pos_1"]),
            )
            readout_2 = _get_readout(
                slug=str(slug),
                sample_id=int(sample_id),
                lang=str(first_row["lang_2"]),
                response=str(first_row["response_2"]),
                full_prompt=str(first_row["full_prompt_2"]),
                pos=int(first_row["pos_2"]),
            )
            all_rows.extend(
                compare_readouts_across_layers(
                    readout_1,
                    readout_2,
                    scenario=str(scenario),
                    sample_id=int(sample_id),
                )
            )
            pbar.update(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    logger.info("Finished recomputing. Total rows: %d saved to %s", len(df), output_path)
    return df


def compute_mean_similarities(
    data: pd.DataFrame | str | Path,
) -> dict[str, pd.DataFrame]:
    """Compute mean cosine similarities and top-K overlaps per scenario and for same vs. different languages."""
    df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()

    if "same_language" not in df.columns:
        if "lang_1" in df.columns and "lang_2" in df.columns:
            df["same_language"] = df["lang_1"] == df["lang_2"]
        else:
            parts = df["scenario"].astype(str).str.split("/", n=1, expand=True)
            df["same_language"] = parts[0] == parts[1]

    agg_spec: dict[str, tuple[str, str]] = {
        "mean": ("cosine_similarity", "mean"),
        "std": ("cosine_similarity", "std"),
        "count": ("cosine_similarity", "count"),
    }
    if "top_k_overlap_count" in df.columns:
        agg_spec["top_k_overlap_mean"] = ("top_k_overlap_count", "mean")
        agg_spec["top_k_overlap_std"] = ("top_k_overlap_count", "std")
    if "raw_top_k_overlap_count" in df.columns:
        agg_spec["raw_top_k_overlap_mean"] = ("raw_top_k_overlap_count", "mean")
        agg_spec["raw_top_k_overlap_std"] = ("raw_top_k_overlap_count", "std")

    by_scenario = df.groupby("scenario", sort=False).agg(**agg_spec).reset_index()
    by_scenario_and_layer = (
        df.groupby(["scenario", "layer"], sort=False).agg(**agg_spec).reset_index()
    )
    by_same_language = (
        df.groupby("same_language", sort=False).agg(**agg_spec).reset_index()
    )
    by_same_language_and_layer = (
        df.groupby(["same_language", "layer"], sort=False).agg(**agg_spec).reset_index()
    )

    return {
        "by_scenario": by_scenario,
        "by_scenario_and_layer": by_scenario_and_layer,
        "by_same_language": by_same_language,
        "by_same_language_and_layer": by_same_language_and_layer,
    }

