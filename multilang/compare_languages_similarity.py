# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Compare transported residual cosine similarity after the </think> block across languages.

Reads two YAML files of :class:`~jlens.examples.Example` instances (e.g. English
and Spanish), pairs examples by identical ``slug``, generates model responses,
extracts Jacobian-transported residuals at the token immediately following the
``</think>`` tag across the specified layers, and records pairwise cosine
similarities for:
    - Language1/Language1
    - Language2/Language2
    - Language1/Language2

All generated prompts (including model answers) and per-layer similarities are
saved to a CSV file, and mean similarities per scenario (and same vs. different
language) are computed via :func:`~jlens.similarity.compute_mean_similarities` and printed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
import transformers

import jlens
from jlens.experiments import (
    compute_mean_similarities,
    recompute_experiment_from_csv,
    run_comparison_experiment,
)
from jlens.generation import default_device, set_seed_n
from jlens.similarity import load_lens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute post-think transported residual cosine similarity and top-K "
            "word overlap across English and Spanish Example YAML files."
        )
    )
    parser.add_argument(
        "--en-yaml",
        type=Path,
        default=None,
        help="Path to the English YAML file containing Example instances (required unless --from-csv is passed).",
    )
    parser.add_argument(
        "--es-yaml",
        type=Path,
        default=None,
        help="Path to the Spanish YAML file containing Example instances (required unless --from-csv is passed).",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing experiment CSV to recompute residuals "
            "and top-K words using its saved full prompts and positions (skips text generation)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g. 'Qwen/Qwen3.5-4B').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for reproducibility (default: 10).",
    )
    parser.add_argument(
        "--lens",
        type=str,
        default=None,
        help=(
            "Path to a custom Jacobian lens .pt file, or a HuggingFace Hub repo ID "
            "(not required if --lens-filename is passed with a custom file)."
        ),
    )
    parser.add_argument(
        "--lens-filename",
        "--lens-file",
        dest="lens_filename",
        type=str,
        default=None,
        help=(
            "Path to a custom Jacobian lens file, or the filename inside the Hub repo "
            "if --lens is a repo ID."
        ),
    )
    parser.add_argument(
        "--lens-revision",
        type=str,
        default=None,
        help="Optional HuggingFace Hub revision if --lens is a repo ID (not needed for custom files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/crosslingual_think_similarity.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of layer indices to evaluate. Defaults to mid layer and "
            "above with a stride of 2, always including the last layer."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top words to extract and compare per layer/position (default: 10).",
    )
    parser.add_argument(
        "--no-mask-display",
        dest="mask_display",
        action="store_false",
        default=True,
        help="Disable meaningful_token_mask filtering when extracting top-K words.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Number of bootstrap comparisons per example and scenario (default: 1).",
    )
    parser.add_argument(
        "--n-single-language",
        type=int,
        default=1,
        help=(
            "Number of single-language comparisons per bootstrap step. Set to 0 (n=0) "
            "to skip single-language comparisons and only compute cross-lingual (English/Spanish) similarity."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3000,
        help="Maximum number of new tokens to generate per prompt (default: 3000).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length when encoding full prompts (default: 4096).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help="Reasoning effort passed to the chat template (default: 'medium').",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking to allow the model response to include thinking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device to load the model on (default: auto-detected cuda/mps/cpu).",
    )
    args = parser.parse_args(argv)
    if not args.from_csv and (not args.en_yaml or not args.es_yaml):
        parser.error(
            "Must provide --en-yaml and --es-yaml (or --from-csv to recompute from an existing CSV)."
        )
    if not args.lens and not args.lens_filename:
        parser.error(
            "Must provide a lens: pass a custom lens file via --lens or --lens-filename, "
            "or a HuggingFace Hub repo via --lens."
        )
    min_required_seq_len = args.max_new_tokens + 512
    if args.max_seq_len < min_required_seq_len:
        args.max_seq_len = min_required_seq_len
    return args


def main(argv: Sequence[str] | None = None) -> None:
    jlens.configure_logging()
    args = parse_args(argv)

    if args.seed is not None:
        set_seed_n(args.seed)

    print(f"Loading model {args.model!r} on {args.device!r}...")
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        ).to(args.device)
    except (ValueError, KeyError):
        hf_model = transformers.AutoModelForConditionalGeneration.from_pretrained(  # type: ignore
            args.model, dtype=torch.bfloat16
        ).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)

    lens = load_lens(
        lens=args.lens,
        lens_filename=args.lens_filename,
        lens_revision=args.lens_revision,
        model_name=args.model,
    )

    if args.from_csv is not None:
        df = recompute_experiment_from_csv(
            model=model,
            lens=lens,
            input_csv_path=args.from_csv,
            output_csv_path=args.output,
            layers=args.layers,
            max_seq_len=args.max_seq_len,
            top_k=args.top_k,
            mask_display=args.mask_display,
        )
    else:
        include_single_lang = args.n_single_language > 0
        df = run_comparison_experiment(
            hf_model=hf_model,
            model=model,
            lens=lens,
            en_yaml_path=args.en_yaml,
            es_yaml_path=args.es_yaml,
            output_csv_path=args.output,
            layers=args.layers,
            n_bootstrap=args.n_bootstrap,
            include_single_language=include_single_lang,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            top_k=args.top_k,
            mask_display=args.mask_display,
        )

    summaries = compute_mean_similarities(df)

    print("\n=== Mean Cosine Similarity & Top-K Overlap by Scenario (Overall) ===")
    print(summaries["by_scenario"].to_string(index=False))

    print("\n=== Mean Cosine Similarity & Top-K Overlap: Same vs. Different Language (Overall) ===")
    print(summaries["by_same_language"].to_string(index=False))

    print("\n=== Mean Cosine Similarity & Top-K Overlap by Scenario and Layer ===")
    print(summaries["by_scenario_and_layer"].to_string(index=False))

    print("\n=== Mean Cosine Similarity & Top-K Overlap: Same vs. Different Language by Layer ===")
    print(summaries["by_same_language_and_layer"].to_string(index=False))


if __name__ == "__main__":
    main()
