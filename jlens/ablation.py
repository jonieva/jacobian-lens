from contextlib import contextmanager
from typing import Any, Generator, List, Union

import torch
import torch.nn.functional as F

from jlens.protocol import LensModel


def get_jacobian_token_direction(
    model: Any, lens: LensModel, layer_idx: int, token_id: int
) -> torch.Tensor:
    """Computes the normalized J-space pullback direction in layer l's residual stream.

    Args:
        model: The language model.
        lens: The fitted Jacobian lens.
        layer_idx: The index of the layer to compute the direction for.
        token_id: The ID of the target token.

    Returns:
        A normalized tensor representing the pullback direction.
    """
    w_u = model.lm_head.weight[token_id].to(torch.float32)
    J_l = lens.jacobians[layer_idx].to(device=w_u.device, dtype=torch.float32)
    direction = J_l.T @ w_u
    return F.normalize(direction, p=2, dim=-1)


@contextmanager
def ablate_jacobian_direction(
    model: Any,
    lens: LensModel,
    token_id: int,
    layers: List[int],
    positions: Union[List[int], slice] = slice(None),
) -> Generator[None, None, None]:
    """Context manager that hooks into model residual streams across specified layers,
    projecting out the J-lens direction for `token_id`.

    Args:
        model: The language model.
        lens: The fitted Jacobian lens.
        token_id: The ID of the token to ablate.
        layers: A list of layer indices to apply the ablation to.
        positions: The sequence positions to apply the ablation to. Defaults to all positions.

    Yields:
        None.
    """
    hooks = []

    # Precompute unit direction vectors for each target layer
    directions = {
        l: get_jacobian_token_direction(model, lens, l, token_id) for l in layers
    }

    def make_hook(layer_idx: int):
        unit_v = directions[layer_idx]

        def hook_fn(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            # Causal LM layers return either a Tensor or a tuple (hidden_states, ...)
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output

            orig_dtype = h.dtype
            h_float = h.to(torch.float32)
            v = unit_v.to(device=h.device, dtype=torch.float32)

            # Compute orthogonal projection: h_proj = (h . v) * v
            # h shape: [batch, seq_len, d_model]
            sub_h = h_float[:, positions, :]
            projection = torch.sum(sub_h * v, dim=-1, keepdim=True) * v

            # Subtract projection from the targeted positions
            h_float[:, positions, :] = sub_h - projection
            h_ablated = h_float.to(orig_dtype)

            if is_tuple:
                return (h_ablated,) + output[1:]
            return h_ablated

        return hook_fn

    # Register forward hooks on the decoder blocks
    # (For Hugging Face Qwen/Llama: model.model.layers or model.transformer.h)
    layer_modules = getattr(model, "model", model).layers
    for l in layers:
        hook = layer_modules[l].register_forward_hook(make_hook(l))
        hooks.append(hook)

    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def swap_jacobian_directions(
    model: Any,
    lens: LensModel,
    source_token_id: int,
    target_token_id: int,
    layers: List[int],
    positions: Union[List[int], slice] = slice(None),
) -> Generator[None, None, None]:
    """Hooks into the specified layers, projecting out the source concept
    and substituting the target concept at matching activation magnitudes.

    Args:
        model: The language model.
        lens: The fitted Jacobian lens.
        source_token_id: The ID of the source token to swap out.
        target_token_id: The ID of the target token to swap in.
        layers: A list of layer indices to apply the swap to.
        positions: The sequence positions to apply the swap to. Defaults to all positions.

    Yields:
        None.
    """
    hooks = []
    device = next(model.parameters()).device

    # Precompute unit direction vectors mapped to target device
    src_dirs = {
        l: get_jacobian_token_direction(model, lens, l, source_token_id).to(device)
        for l in layers
    }
    tgt_dirs = {
        l: get_jacobian_token_direction(model, lens, l, target_token_id).to(device)
        for l in layers
    }

    def make_hook(layer_idx: int):
        v_s = src_dirs[layer_idx]
        v_t = tgt_dirs[layer_idx]

        def hook_fn(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output

            orig_dtype = h.dtype
            h_float = h.to(torch.float32)

            # Isolate the targeted sequence positions (slice(None) handles KV-cache generation safely)
            sub_h = h_float[:, positions, :]

            # 1. Measure the active magnitude of the source concept
            proj_mag = torch.sum(sub_h * v_s, dim=-1, keepdim=True)

            # 2. Subtract the source direction, add the target direction
            h_swapped = sub_h - (proj_mag * v_s) + (proj_mag * v_t)

            h_float[:, positions, :] = h_swapped
            h_out = h_float.to(orig_dtype)

            return (h_out,) + output[1:] if is_tuple else h_out

        return hook_fn

    # Register forward hooks on the decoder blocks
    layer_modules = getattr(model, "model", model).layers
    for l in layers:
        hook = layer_modules[l].register_forward_hook(make_hook(l))
        hooks.append(hook)

    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()