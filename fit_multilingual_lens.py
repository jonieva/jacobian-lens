# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fit a multilingual Jacobian lens for a model using multilingual corpora.

Loads multilingual datasets (e.g. Wikimedia Wikipedia in multiple languages and/or
Salesforce WikiText), extracts prompts, and computes the Jacobian transport matrices
via :func:`~jlens.fit`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset

import jlens
from jlens.examples import load_wikitext_prompts
from jlens.generation import default_device, set_seed_n


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a multilingual Jacobian lens for a specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g. 'Qwen/Qwen3.5-4B').",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "es"],
        help="Languages to include in the corpus (default: ['en', 'es']).",
    )
    parser.add_argument(
        "--n-prompts-per-lang",
        type=int,
        default=50,
        help="Number of prompts to sample per language (default: 50).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikimedia/wikipedia",
        help="Base HuggingFace dataset name (default: 'wikimedia/wikipedia').",
    )
    parser.add_argument(
        "--dataset-date",
        type=str,
        default="20231101",
        help="Wikipedia snapshot date tag (default: '20231101').",
    )
    parser.add_argument(
        "--use-wikitext-for-en",
        action="store_true",
        help="Use 'Salesforce/wikitext' ('wikitext-103-raw-v1') for English instead of Wikipedia.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Source layers to fit. Defaults to all layers below target_layer (unless --stride is passed).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Optional stride for layer selection (e.g. --stride 2). When set (and --layers is not passed), "
            "selects layers from the middle layer up to target_layer with this stride, matching "
            "compare_languages_similarity.py."
        ),
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="Target layer to differentiate towards (default: final layer).",
    )
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=None,
        help="Output dimensions computed per backward pass (default: 8 on CUDA, 1 on MPS/CPU).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length per prompt for Jacobian estimation (default: 128).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=600,
        help="Minimum character length to accept a raw text record (default: 600).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for data sampling and reproducibility (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the fitted .pt lens file. Defaults to "
            "'data/computed_lens/<model_name>/<model_name>_<langs>_lens.pt'."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path to write / resume fitting checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save checkpoint every N prompts (default: 200).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing checkpoint_path if found.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device to load the model on (default: auto-detected cuda/mps/cpu).",
    )
    return parser.parse_args(argv)


def load_multilingual_prompts(
    languages: Sequence[str],
    n_prompts_per_lang: int,
    *,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_date: str = "20231101",
    use_wikitext_for_en: bool = False,
    min_chars: int = 600,
    seed: int = 10,
) -> tuple[list[str], dict[str, int]]:
    """Load and mix prompts evenly across specified languages."""
    prompts_by_lang: dict[str, list[str]] = {}

    for lang in languages:
        lang_code = lang.strip().lower()
        if lang_code == "en" and use_wikitext_for_en:
            print(f"Streaming English prompts from Salesforce/wikitext...")
            ds = load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split="train",
                streaming=True,
            )
        else:
            config_name = f"{dataset_date}.{lang_code}"
            print(f"Streaming {lang_code!r} prompts from {dataset_name} ({config_name})...")
            ds = load_dataset(dataset_name, config_name, split="train", streaming=True)

        ds = ds.shuffle(seed=seed, buffer_size=1000)
        lang_prompts = load_wikitext_prompts(
            n_prompts=n_prompts_per_lang,
            dataset=ds,
            min_chars=min_chars,
        )
        print(f"  Loaded {len(lang_prompts)} prompts for language {lang_code!r}")
        prompts_by_lang[lang_code] = lang_prompts

    combined_prompts: list[str] = []
    counts: dict[str, int] = {}
    for lang, p_list in prompts_by_lang.items():
        combined_prompts.extend(p_list)
        counts[lang] = len(p_list)

    random.seed(seed)
    random.shuffle(combined_prompts)
    return combined_prompts, counts


def main(argv: Sequence[str] | None = None) -> None:
    jlens.configure_logging()
    args = parse_args(argv)

    if args.seed is not None:
        set_seed_n(args.seed)

    model_short_name = args.model.split("/")[-1]
    lang_summary = "_".join(f"{args.n_prompts_per_lang}{lang}" for lang in args.languages)

    if args.output is None:
        folder = Path(f"data/computed_lens/{model_short_name}")
        output_file = folder / f"{model_short_name}_{lang_summary}_lens.pt"
    else:
        output_file = Path(args.output)
        folder = output_file.parent

    folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model!r} on {args.device!r}...")
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        ).to(args.device)
    except (ValueError, KeyError):
        hf_model = transformers.AutoModelForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16
        ).to(args.device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)

    dim_batch = args.dim_batch
    if dim_batch is None:
        dim_batch = 8 if torch.cuda.is_available() else 1

    print(f"Sampling {args.n_prompts_per_lang} prompts per language across {args.languages}...")
    prompts, counts = load_multilingual_prompts(
        languages=args.languages,
        n_prompts_per_lang=args.n_prompts_per_lang,
        dataset_name=args.dataset,
        dataset_date=args.dataset_date,
        use_wikitext_for_en=args.use_wikitext_for_en,
        min_chars=args.min_chars,
        seed=args.seed,
    )

    if not prompts:
        raise RuntimeError("No prompts were loaded! Please verify the language codes and dataset.")

    target_layer = (
        args.target_layer if args.target_layer is not None else model.n_layers - 1
    )
    if args.layers is not None:
        source_layers = args.layers
    elif args.stride is not None:
        from jlens.experiments import resolve_default_layers
        candidate_layers = resolve_default_layers(model.n_layers, stride=args.stride)
        source_layers = [l for l in candidate_layers if l < target_layer]
    else:
        source_layers = None

    print(
        f"Fitting Jacobian lens over {len(prompts)} total prompts "
        f"(layers={source_layers if source_layers is not None else 'all'}, "
        f"target_layer={target_layer}, dim_batch={dim_batch}, max_seq_len={args.max_seq_len})..."
    )

    checkpoint_str = str(args.checkpoint_path) if args.checkpoint_path else None
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=checkpoint_str,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )

    lens.save(str(output_file))
    print(f"\n[SUCCESS] Jacobian lens successfully saved to {output_file}")

    # Generate metadata README alongside the lens
    readme_path = folder / "README.md"
    readme_content = (
        f"# Configuration for {model_short_name} Multilingual Jacobian Lens\n\n"
        f"- **Model**: `{args.model}`\n"
        f"- **Languages**: `{', '.join(args.languages)}`\n"
        f"- **Prompts per language**: `{args.n_prompts_per_lang}` (Total: `{len(prompts)}`)\n"
        f"- **Prompt counts by language**: `{counts}`\n"
        f"- **Dataset**: `{args.dataset}` (date `{args.dataset_date}`)\n"
        f"- **Fitted Layers**: `{source_layers if source_layers is not None else 'all'}`\n"
        f"- **Target Layer**: `{target_layer}`\n"
        f"- **Max Seq Len**: `{args.max_seq_len}`\n"
        f"- **Dim Batch**: `{dim_batch}`\n"
        f"- **Seed**: `{args.seed}`\n"
        f"- **Saved Lens File**: `{output_file.name}`\n"
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Metadata recorded in {readme_path}")


if __name__ == "__main__":
    main()

