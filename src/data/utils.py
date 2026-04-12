"""
Batch collation and trajectory splitting utilities for SSM datasets.

Conventions:
- Input items must use keys: `x_traj`, `x_tps`, `x_mask`.
- Timestamps can be irregular and can differ across samples before collation.
- Returned batched dictionaries use `x_*` for context and optional `y_*` for
  targets/extrapolation.
- `x_tps` / `y_tps` returned by these collators are shared 1D grids
  (`[T]`), not per-sample batched grids (`[B, T]`).
"""

import torch


def _extract_traj_item(item: dict):
    """
    Extract canonical trajectory fields from one dataset sample.

    Args:
        item: Sample dictionary containing `x_traj`, `x_tps`, `x_mask`.

    Returns:
        Tuple `(traj, tps, mask)` where:
        - `traj`: `[T_i, dim_x]`
        - `tps`: `[T_i]`
        - `mask`: `[T_i, dim_x]`
    """
    if "x_traj" in item and "x_tps" in item and "x_mask" in item:
        return item["x_traj"], item["x_tps"], item["x_mask"]
    raise KeyError("Expected keys ('x_traj','x_tps','x_mask').")


def _stack_if_shared_timeline(
    batch_data: list[dict],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
):
    """
    Fast path for batches that already share one timeline up to float tolerance.

    This is especially important for windowed datasets with relative float32
    timestamps, where mathematically identical grids can differ by tiny rounding
    noise across samples. In that case, building an exact-value union timeline
    would create many near-duplicate timestamps and spuriously sparse batches.

    Returns:
        `(data, tps, mask)` when all samples share one timeline and can be
        stacked directly, otherwise `None`.
    """
    if not batch_data:
        raise ValueError("batch_data must be non-empty")

    ref_traj, ref_tps, ref_mask = _extract_traj_item(batch_data[0])
    traj_all = [ref_traj]
    mask_all = [ref_mask]

    for item in batch_data[1:]:
        traj, tps, mask = _extract_traj_item(item)
        if traj.shape != ref_traj.shape or mask.shape != ref_mask.shape or tps.shape != ref_tps.shape:
            return None
        if not torch.allclose(tps, ref_tps, atol=atol, rtol=rtol):
            return None
        traj_all.append(traj)
        mask_all.append(mask)

    return (
        torch.stack(traj_all, dim=0),
        ref_tps.clone(),
        torch.stack(mask_all, dim=0),
    )


def batch_data_alignment(batch_data: list[dict]):
    """
    Align an irregular batch onto a shared timestamp grid.

    Each sample may have different timestamps. This function:
    1) builds the sorted union of timestamps across the batch;
    2) places each sample's observations at matching union indices;
    3) fills missing values with zeros and marks them in `mask_aligned`.

    Args:
        batch_data: List of sample dicts containing:
            - `x_traj`: `[T_i, dim_x]`
            - `x_tps`: `[T_i]`
            - `x_mask`: `[T_i, dim_x]`

    Returns:
        data_aligned: Tensor `[B, T_union, dim_x]`
        union_t: Tensor `[T_union]`
        mask_aligned: Tensor `[B, T_union, dim_x]`
    """
    stacked = _stack_if_shared_timeline(batch_data)
    if stacked is not None:
        return stacked

    union_t, inverse_indices = torch.unique(
        torch.cat([_extract_traj_item(data_time_mask_dict)[1] for data_time_mask_dict in batch_data]),
        sorted=True,
        return_inverse=True,
    )
    union_seq_len = union_t.shape[0]
    batch_size, n_variables = len(batch_data), _extract_traj_item(batch_data[0])[0].shape[-1]

    data_aligned = torch.zeros((batch_size, union_seq_len, n_variables), dtype=torch.float32)
    mask_aligned = torch.zeros_like(data_aligned, dtype=torch.float32)

    offset = 0
    for i, item in enumerate(batch_data):
        traj, tps, mask = _extract_traj_item(item)
        indices = inverse_indices[offset: (offset + len(tps))]
        data_aligned[i, indices, :] = traj
        mask_aligned[i, indices, :] = mask
        offset += len(tps)
    
    return data_aligned, union_t, mask_aligned


def split_traj(data: torch.tensor, timestamps: torch.tensor, mask, cutoff_len=None) -> dict:
    """
    Split aligned trajectories into context (`x_*`) and optional target (`y_*`).

    Args:
        data: Aligned observations `[B, T, dim_x]`.
        timestamps: Shared timestamps `[T]`.
        mask: Aligned mask `[B, T, dim_x]`.
        cutoff_len:
            - `None`: return context-only (`x_*`) over full length;
            - `"default"`: split at `T // 2`;
            - `int`: explicit context length, must satisfy `0 < cutoff_len < T`.

    Returns:
        Dict containing:
        - always: `x_traj`, `x_tps`, `x_mask`
        - when split: `y_traj`, `y_tps`, `y_mask`

        Shape convention in this module:
        - `x_traj`, `x_mask`: `[B, T, dim_x]`
        - `x_tps`: `[T]` (shared timeline)
        - `y_tps`: `[T_pred]` when present (shared timeline)
    """
    assert cutoff_len is None or isinstance(cutoff_len, int) or cutoff_len == 'default', "cutoff_len should be an integer or None or `default`"
    if cutoff_len is None:
        return {
            'x_traj': data,
            'x_tps': timestamps,
            'x_mask': mask,
        }
    elif cutoff_len == 'default':
        cutoff_len = data.shape[1] // 2
    else:
        assert cutoff_len < data.shape[1], f"cutoff_len should be strictly smaller than sequence length ({data.shape[1]})"

    return {
        'x_traj': data[:, :cutoff_len, :],
        'x_tps': timestamps[:cutoff_len],
        'x_mask': mask[:, :cutoff_len, :],
        'y_traj': data[:, cutoff_len:, :],
        'y_tps': timestamps[cutoff_len:],
        'y_mask': mask[:, cutoff_len:, :]
    }


def prepare_batched_traj_extrap(batch_data: list[dict], tps_extrap: torch.tensor) -> dict:
    """
    Build a batched context dictionary for extrapolation tasks.

    This aligns irregular samples onto a union timestamp grid and keeps the
    full aligned sequence as context (`x_*`). Extrapolation timestamps are
    attached as `y_tps`; model predictions at `y_tps` can then be compared
    externally.

    Args:
        batch_data: List of sample dicts with `x_traj/x_tps/x_mask`.
        tps_extrap: Extrapolation timeline `[T_pred]`.

    Returns:
        Batched dict with:
        - `x_traj`: `[B, T_ctx, dim_x]`
        - `x_mask`: `[B, T_ctx, dim_x]`
        - `x_tps`: `[T_ctx]` (shared context timeline)
        - `y_tps`: `[T_pred]` (shared prediction timeline)
    """
    data_aligned, union_t, mask_aligned = batch_data_alignment(batch_data)
    batched_data = split_traj(data_aligned, union_t, mask_aligned, cutoff_len=None)
    batched_data['y_tps'] = tps_extrap.clone()
    return batched_data


def prepare_batched_traj(batch_data: list[dict], cutoff_len: str|int=None) -> dict:
    """
    Collate function for reconstruction/interpolation-style batching.

    Args:
        batch_data: List of sample dicts with `x_traj/x_tps/x_mask`.
        cutoff_len:
            - `None`: no split; reconstruction mode (`x_*` only);
            - `"default"` or `int`: split into context (`x_*`) and target (`y_*`).

    Returns:
        Batched dict produced by `split_traj(...)`, where `x_tps`/`y_tps`
        are shared 1D timelines.
    """
    data_aligned, union_t, mask_aligned = batch_data_alignment(batch_data)
    return split_traj(data_aligned, union_t, mask_aligned, cutoff_len=cutoff_len)
