# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Model response generation, prompt formatting, and thinking block detection.

Provides:
- Device and seeding helpers (:func:`default_device`, :func:`set_seed_n`)
- Dynamic inspection of chat-template thinking delimiters (:func:`get_thinking_template_spec`, :class:`ThinkingTemplateSpec`)
- Position finding for thinking blocks and prompt words (:func:`get_token_position_after_think_block`, :func:`find_word_position_in_example`)
- Prompt resolution (:func:`resolve_prompt_extended`)
- Model text generation with interrupted-thinking handling (:func:`get_model_response`)
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers

from jlens.examples import Example, resolve_prompt

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Device & Seeding Helpers
# --------------------------------------------------------------------------- #


def default_device() -> str:
    """Return the preferred PyTorch device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed_n(seed: int = 10) -> None:
    """Set random seed across Python, NumPy, PyTorch (CPU/CUDA/MPS), and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


# --------------------------------------------------------------------------- #
# Dynamic Thinking Template Delimiters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ThinkingTemplateSpec:
    """Delimiters for a model's thinking block, inferred directly from its tokenizer chat template."""

    supports_thinking: bool
    open_tag: str = ""
    close_tag: str = ""
    close_suffix: str = ""


_THINK_SPEC_CACHE: dict[int, ThinkingTemplateSpec] = {}


def get_thinking_template_spec(tokenizer: Any) -> ThinkingTemplateSpec:
    """Infer thinking block tags (e.g. ``<think>``/``</think>`` or ``<|thought|>``/``<|/thought|>``)
    directly from ``tokenizer.apply_chat_template`` without hardcoding model-specific strings.
    """
    cache_key = id(tokenizer)
    if cache_key in _THINK_SPEC_CACHE:
        return _THINK_SPEC_CACHE[cache_key]

    if getattr(tokenizer, "chat_template", None) is None:
        spec = ThinkingTemplateSpec(supports_thinking=False)
        _THINK_SPEC_CACHE[cache_key] = spec
        return spec

    think_marker = "__JLENS_THINK_MARKER__"
    content_marker = "__JLENS_CONTENT_MARKER__"
    probe_messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "thinking": think_marker,
            "reasoning_content": think_marker,
            "content": content_marker,
        },
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            probe_messages, tokenize=False, continue_final_message=True
        )
    except Exception:
        spec = ThinkingTemplateSpec(supports_thinking=False)
        _THINK_SPEC_CACHE[cache_key] = spec
        return spec

    if think_marker not in rendered or content_marker not in rendered:
        spec = ThinkingTemplateSpec(supports_thinking=False)
        _THINK_SPEC_CACHE[cache_key] = spec
        return spec

    # Slice out the transition placed between thought and final content
    think_end_idx = rendered.index(think_marker) + len(think_marker)
    content_start_idx = rendered.index(content_marker)
    close_suffix = rendered[think_end_idx:content_start_idx]
    close_tag = close_suffix.strip()

    # Slice out the opening tag preceding the thought marker
    prefix_before_think = rendered[: rendered.index(think_marker)]
    try:
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        common_len = 0
        for a, b in zip(prefix_before_think, user_only):
            if a != b:
                break
            common_len += 1
        open_tag = prefix_before_think[common_len:].strip()
    except Exception:
        open_tag = ""

    if not open_tag and close_tag.startswith("</"):
        open_tag = "<" + close_tag[2:]

    spec = ThinkingTemplateSpec(
        supports_thinking=bool(close_tag),
        open_tag=open_tag,
        close_tag=close_tag,
        close_suffix=close_suffix,
    )
    _THINK_SPEC_CACHE[cache_key] = spec
    return spec


def resolve_prompt_extended(
    example: Example,
    tokenizer: Any,
    enable_thinking: bool = True,
    reasoning_effort: str = "medium",
) -> str:
    """Return the formatted prompt string for ``example`` using the tokenizer's chat template.

    Falls back to raw prompt text when ``example.user`` is ``None``.
    When ``enable_thinking`` is True and the model supports thinking delimiters, ensures
    the prompt ends with the opening thinking tag (e.g. ``<think>`` or ``<|channel>thought``)
    so the model is forced into thinking mode.
    """
    prompt = resolve_prompt(
        example,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    if enable_thinking:
        spec = get_thinking_template_spec(tokenizer)
        if spec.supports_thinking and spec.open_tag:
            if not prompt.rstrip().endswith(spec.open_tag):
                if not prompt.endswith("\n"):
                    prompt += "\n"
                prompt += f"{spec.open_tag}\n"
    return prompt


def get_token_position_after_think_block(
    tokenizer: Any,
    prompt: str,
    *,
    base_prompt: str | None = None,
    max_seq_len: int | None = None,
) -> int:
    """Return the token index right where the final answer is predicted (the token
    immediately preceding the first non-whitespace answer token after the closing thinking
    tag, or at the end of ``base_prompt`` for models without thinking tags).

    Args:
        tokenizer: HuggingFace tokenizer instance.
        prompt: Full prompt string (including any thinking block and response).
        base_prompt: Optional input prompt string before generation; used as a clean
            fallback boundary for models whose chat template has no thinking block.
        max_seq_len: Optional maximum sequence length.

    Returns:
        Token index that predicts the first answer token, or ``-1`` if not found.
    """
    spec = get_thinking_template_spec(tokenizer)
    closing_tag = spec.close_tag or "</think>"

    if closing_tag in prompt:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        full_tokens: list[int] = encoded.input_ids[0].tolist()
        tag_tokens: list[int] = tokenizer.encode(closing_tag, add_special_tokens=False)
        if tag_tokens:
            tag_len = len(tag_tokens)
            for i in range(len(full_tokens) - tag_len + 1):
                if full_tokens[i : i + tag_len] == tag_tokens:
                    pos_after = i + tag_len
                    # Find the first token with non-whitespace text at or after pos_after
                    for k in range(pos_after, len(full_tokens)):
                        decoded = tokenizer.decode(
                            [full_tokens[k]], clean_up_tokenization_spaces=False
                        )
                        if decoded.strip():
                            return k - 1
                    return max(0, pos_after - 1)

    # Fallback for models without thinking tags (or when thinking tag wasn't output):
    # the answer starts immediately after base_prompt, so the token immediately preceding
    # the first non-whitespace answer token predicts the answer.
    if base_prompt is not None:
        base_tokens = tokenizer(
            base_prompt, return_tensors="pt", truncation=False
        ).input_ids[0].tolist()
        full_tokens = tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).input_ids[0].tolist()
        if 0 < len(base_tokens) < len(full_tokens):
            pos_after = len(base_tokens)
            for k in range(pos_after, len(full_tokens)):
                decoded = tokenizer.decode(
                    [full_tokens[k]], clean_up_tokenization_spaces=False
                )
                if decoded.strip():
                    return k - 1
            return len(base_tokens) - 1

    return -1


def find_word_position_in_example(
    example: Example,
    target_substring: str,
    tokenizer: Any,
    *,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
    max_seq_len: int = 4000,
    verbose: bool = True,
) -> list[int]:
    """Find all token positions overlapping ``target_substring`` in ``example`` after chat formatting.

    Args:
        example: The :class:`~jlens.examples.Example` containing the prompt.
        target_substring: Substring to locate inside ``example.user`` (or ``example.prompt``).
        tokenizer: HuggingFace tokenizer.
        enable_thinking: Passed to :func:`resolve_prompt_extended`.
        reasoning_effort: Passed to :func:`resolve_prompt_extended`.
        max_seq_len: Maximum sequence length for tokenization.
        verbose: If ``True``, prints matched token positions and strings.

    Returns:
        List of all token indices overlapping ``target_substring``.
    """
    full_prompt = resolve_prompt_extended(
        example,
        tokenizer,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    body_text = example.user if example.user is not None else example.prompt
    if body_text is None:
        raise ValueError(f"Example {example.slug!r} has neither user nor prompt text")

    body_offset = full_prompt.find(body_text)
    if body_offset == -1:
        raise ValueError("Could not locate example body inside formatted chat prompt")

    char_spans: list[tuple[int, int]] = []
    start_idx = 0
    while True:
        found = body_text.find(target_substring, start_idx)
        if found == -1:
            break
        c_start = body_offset + found
        char_spans.append((c_start, c_start + len(target_substring)))
        start_idx = found + 1

    if not char_spans:
        raise ValueError(
            f"Substring {target_substring!r} not found in example {example.slug!r}: {body_text!r}"
        )

    encoded = tokenizer(
        full_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
    )
    token_ids: list[int] = encoded.input_ids[0].tolist()
    ctx_strs: list[str] = [
        tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids
    ]

    matched_positions: list[int] = []
    cur_char = 0
    for pos, tok_str in enumerate(ctx_strs):
        tok_start = cur_char
        tok_end = cur_char + len(tok_str)
        cur_char = tok_end
        if any(max(tok_start, cs) < min(tok_end, ce) for cs, ce in char_spans):
            matched_positions.append(pos)

    if not matched_positions:
        raise ValueError(
            f"No tokens overlapped character spans {char_spans} for {target_substring!r}"
        )

    if verbose:
        matched_strs = [ctx_strs[p] for p in matched_positions]
        logger.info(
            "[%s] %r -> positions %s (%s)",
            example.slug,
            target_substring,
            matched_positions,
            matched_strs,
        )
    return matched_positions


# --------------------------------------------------------------------------- #
# Model Response Generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def get_model_responses(
    hf_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    n: int = 1,
    max_tokens: int = 512,
    forced_answer_tokens: int = 100,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
    enable_thinking: bool | None = None,
) -> list[str]:
    """Generate ``n`` model responses for ``prompt`` using sampling in a single batched call,
    with automatic interrupted-thinking continuation."""
    if n <= 0:
        return []
    spec = get_thinking_template_spec(tokenizer)
    if enable_thinking and spec.supports_thinking and spec.open_tag:
        if not prompt.rstrip().endswith(spec.open_tag):
            if not prompt.endswith("\n"):
                prompt += "\n"
            prompt += f"{spec.open_tag}\n"

    device = next(hf_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    output = hf_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_return_sequences=n,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    responses: list[str] = []
    for i in range(n):
        response_ids = output[i, prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

        # If the model opened a thinking block that didn't close within max_tokens,
        # force-close using the template's own closing suffix and generate the final answer.
        if spec.supports_thinking and spec.close_tag:
            has_open_think = (
                not spec.open_tag or spec.open_tag in prompt or spec.open_tag in response_text
            ) and (spec.close_tag not in prompt and spec.close_tag not in response_text)
            if has_open_think:
                logger.info("Thinking interrupted at %d tokens; force-closing...", max_tokens)
                forced_suffix = spec.close_suffix or f"\n{spec.close_tag}\n\n"
                forced_prompt = prompt + response_text + forced_suffix
                forced_inputs = tokenizer(forced_prompt, return_tensors="pt").to(device)
                forced_output = hf_model.generate(
                    **forced_inputs,
                    max_new_tokens=forced_answer_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                answer_ids = forced_output[0, forced_inputs.input_ids.shape[1] :]
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
                response_text = response_text + forced_suffix + answer_text

        responses.append(response_text)

    return responses


@torch.no_grad()
def get_model_response(
    hf_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 512,
    forced_answer_tokens: int = 100,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
    enable_thinking: bool | None = None,
) -> str:
    """Generate a single model response for ``prompt`` using sampling, with automatic interrupted-thinking continuation."""
    return get_model_responses(
        hf_model,
        tokenizer,
        prompt,
        n=1,
        max_tokens=max_tokens,
        forced_answer_tokens=forced_answer_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
    )[0]

