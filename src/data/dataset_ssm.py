from __future__ import annotations
"""
Dataset utilities for latent state-space time-series modeling.

PUBLIC API
----------
- `TSDataset`: partitions one long time series into fixed-length windows.
- `TrajDataset`: consumes already segmented trajectories.

Both return dictionaries with keys: `x_traj`, `x_mask`, `x_tps` (context)
and optionally `y_traj`, `y_mask`, `y_tps` (prediction target when pred_len > 0).

Timestamps are converted to float. If `relative_timestamps=True`, 
timestamps are shifted so context starts at zero.

Timestamp shape note:
- Per-sample dataset outputs use 1D timestamps (`x_tps: [L_ctx]`, `y_tps: [L_pred]`).
- With default PyTorch collation, batch timestamps become batched tensors
  (`x_tps: [B, L_ctx]`, `y_tps: [B, L_pred]`).
- With custom collators in `src.data.utils` (e.g. `prepare_batched_traj*`),
  timestamps are returned as shared 1D grids.

INTERNAL HELPERS
----------------
- `DataPack`: data normalization container (not for direct use)
- `_BaseTrajectoryDataset`: shared base class (not for direct use)
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset


# =============================================================================
# Constants
# =============================================================================

UNK = 0.0


# =============================================================================
# Helper Classes (Internal)
# =============================================================================

@dataclass(frozen=True)
class DataPack:
    """
    Helper class: Standardized container for trajectory-ready arrays.

    This is an internal data container used by TSDataset and TrajDataset
    to normalize and validate input data before partitioning into samples.

    Attributes:
        data: Dense observation array of shape `[L, dim_x]`.
        mask: Observation mask array of shape `[L, dim_x]`, where 1 indicates
            observed/valid values and 0 indicates missing values.
        ts: Timestamp array of shape `[L]`.
    """

    data: np.ndarray  # [L, dim_x]
    mask: np.ndarray  # [L, dim_x]
    ts: np.ndarray    # [L]

    @staticmethod
    def _default_observation_mask(x: np.ndarray) -> np.ndarray:
        """
        Build the default observation mask.

        Behavior:
          - if all entries are finite, return an all-ones mask;
          - otherwise, return finite-value indicator (1 for finite, 0 otherwise).
        """
        if np.isfinite(x).all():
            return np.ones_like(x, dtype=np.int8)
        return np.isfinite(x).astype(np.int8)

    @classmethod
    def from_time_series(
        cls,
        data: pd.DataFrame | np.ndarray,
        fillna: float | str | None = UNK,
        mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        ts_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray] | None = None,
    ) -> "DataPack":
        """
        Build a `DataPack` from one unsegmented time series.

        Args:
            data: Either:
                - `pd.DataFrame` with shape `[L, dim_x]`, index used as default timestamps.
                - `np.ndarray` with shape `[L, dim_x]` (or `[L]` for univariate).
            fillna:
                - numeric value: fill all NaNs with that value;
                - `"ffill"`: forward-fill then fill remaining NaNs with `UNK`;
                - `None`: keep NaNs as-is.
            mask_fn: Optional function mapping raw data -> mask with shape
                compatible with data. Default:
                - all ones when no missing/non-finite entries exist,
                - finite-value indicator otherwise.
            ts_fn: Optional function mapping input data -> timestamps.
                Default:
                - DataFrame: `data.index.to_numpy()`
                - ndarray: `np.arange(L)`

        Returns:
            A normalized `DataPack` with dense `data`, `mask`, and `ts`.
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("data must be a pandas.DataFrame or numpy.ndarray")

        if mask_fn is None:
            def mask_fn(x: np.ndarray) -> np.ndarray:
                return cls._default_observation_mask(x)

        if ts_fn is None:
            def ts_fn(d: pd.DataFrame | np.ndarray) -> np.ndarray:
                if isinstance(d, pd.DataFrame):
                    return d.index.to_numpy()
                return np.arange(d.shape[0])

        raw = np.asarray(data.values if isinstance(data, pd.DataFrame) else data)
        mask = mask_fn(raw)
        ts = np.asarray(ts_fn(data))

        x_df = pd.DataFrame(raw)
        if fillna == "ffill":
            x_df = x_df.ffill().fillna(UNK)
        elif fillna is not None:
            x_df = x_df.fillna(fillna)

        x = x_df.values
        if x.ndim == 1:
            x = x[:, None]
        # Keep mask shape aligned with data shape, especially for univariate
        # ndarray inputs where DataFrame coercion makes x 2D but mask may remain 1D.
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.shape != x.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match data shape {x.shape} after normalization."
            )
        return cls(data=np.asarray(x), mask=np.asarray(mask), ts=ts)

    @classmethod
    def from_trajectory(
        cls,
        traj,
        tps,
        mask=None,
        fillna: float | str | None = UNK,
        mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "DataPack":
        """
        Build a `DataPack` from one already segmented trajectory.

        Args:
            traj: Observation trajectory with shape `[L, dim_x]` or `[L]`.
            tps: Timestamps with shape `[L]`.
            mask: Optional mask of shape `[L, dim_x]` or `[L]`.
                If omitted, defaults to:
                - all ones when no missing/non-finite entries exist,
                - finite-value indicator otherwise
                (or `mask_fn` result when provided).
            fillna:
                - numeric value: fill NaNs;
                - `"ffill"`: forward-fill then fill remaining NaNs with `UNK`;
                - `None`: keep NaNs.
            mask_fn: Optional mask constructor when `mask` is not provided.

        Returns:
            A normalized `DataPack` with dense `data`, `mask`, and `ts`.
        """
        x = np.asarray(traj)
        if x.ndim == 1:
            x = x[:, None]
        if mask is None:
            if mask_fn is None:
                mask = cls._default_observation_mask(x)
            else:
                mask = mask_fn(x)
        else:
            mask = np.asarray(mask)
            if mask.ndim == 1:
                mask = mask[:, None]

        x_df = pd.DataFrame(x)
        if fillna == "ffill":
            x_df = x_df.ffill().fillna(UNK)
        elif fillna is not None:
            x_df = x_df.fillna(fillna)
        x = x_df.values

        ts = np.asarray(tps)
        return cls(data=np.asarray(x), mask=np.asarray(mask), ts=ts)


class _BaseTrajectoryDataset(Dataset):
    """
    Base class for trajectory datasets (helper - not for direct use).

    This class is responsible for:
    - timestamp normalization/conversion (`_to_float_ts`);
    - appending normalized context/target samples (`_append_sample`);
    - providing a consistent `__getitem__` dictionary schema.
    
    Use `TSDataset` or `TrajDataset` instead of instantiating this directly.
    """

    def __init__(self, pred_len: int = 0, relative_timestamps: bool = True):
        self.pred_len = int(pred_len)
        self.relative_timestamps = bool(relative_timestamps)
        # Internal storage: each element stores one sample
        self._x_data: list[np.ndarray] = []  # each: [context_len, dim_x]
        self._x_mask: list[np.ndarray] = []  # each: [context_len, dim_x]
        self._x_tps: list[np.ndarray] = []   # each: [context_len]
        self._y_data: list[np.ndarray] = []  # each: [pred_len, dim_x]
        self._y_mask: list[np.ndarray] = []  # each: [pred_len, dim_x]
        self._y_tps: list[np.ndarray] = []   # each: [pred_len]

    @staticmethod
    def _to_float_ts(ts: np.ndarray) -> np.ndarray:
        """
        Convert timestamps to floating-point values.

        Datetime arrays are converted to nanosecond integers, then cast to
        float. Numeric timestamps are cast directly to float.
        """
        ts = np.asarray(ts)
        if np.issubdtype(ts.dtype, np.datetime64):
            return ts.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
        return ts.astype(np.float64)

    def _append_sample(self, x, x_mask, x_tps, y=None, y_mask=None, y_tps=None) -> None:
        """
        Append one normalized sample to internal dataset buffers.

        Args:
            x: [context_len, dim_x] - Context observations.
            x_mask: [context_len, dim_x] - Context mask.
            x_tps: [context_len] - Context timestamps.
            y: [pred_len, dim_x] - Optional prediction observations.
            y_mask: [pred_len, dim_x] - Optional prediction mask.
            y_tps: [pred_len] - Optional prediction timestamps.
        """
        x_tps = self._to_float_ts(x_tps)
        if self.relative_timestamps:
            t0 = x_tps[0]
            x_tps = x_tps - t0
            if y_tps is not None:
                y_tps = self._to_float_ts(y_tps) - t0
        else:
            if y_tps is not None:
                y_tps = self._to_float_ts(y_tps)

        # Store as float32 arrays
        self._x_data.append(np.asarray(x, dtype=np.float32))        # [context_len, dim_x]
        self._x_mask.append(np.asarray(x_mask, dtype=np.float32))   # [context_len, dim_x]
        self._x_tps.append(np.asarray(x_tps, dtype=np.float32))     # [context_len]
        if y is not None:
            self._y_data.append(np.asarray(y, dtype=np.float32))    # [pred_len, dim_x]
            self._y_mask.append(np.asarray(y_mask, dtype=np.float32))  # [pred_len, dim_x]
            self._y_tps.append(np.asarray(y_tps, dtype=np.float32))  # [pred_len]

    def __len__(self) -> int:
        return len(self._x_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one sample as torch tensors.

        Returns:
            dict with:
            - 'x_traj': [context_len, dim_x] - context observations (always present)
            - 'x_mask': [context_len, dim_x] - context mask (always present)
            - 'x_tps': [context_len] - context timestamps (always present)
            - 'y_traj': [pred_len, dim_x] - prediction observations (if pred_len > 0)
            - 'y_mask': [pred_len, dim_x] - prediction mask (if pred_len > 0)
            - 'y_tps': [pred_len] - prediction timestamps (if pred_len > 0)

        Notes:
            This method returns one sample. Batched shapes depend on DataLoader
            collation (default collation vs custom `collate_fn`).
        """
        out = {
            "x_traj": torch.tensor(self._x_data[idx], dtype=torch.float32),  # [context_len, dim_x]
            "x_mask": torch.tensor(self._x_mask[idx], dtype=torch.float32),  # [context_len, dim_x]
            "x_tps": torch.tensor(self._x_tps[idx], dtype=torch.float32),    # [context_len]
        }
        if self.pred_len > 0:
            out["y_traj"] = torch.tensor(self._y_data[idx], dtype=torch.float32)  # [pred_len, dim_x]
            out["y_mask"] = torch.tensor(self._y_mask[idx], dtype=torch.float32)  # [pred_len, dim_x]
            out["y_tps"] = torch.tensor(self._y_tps[idx], dtype=torch.float32)    # [pred_len]
        return out


# =============================================================================
# Public API Classes
# =============================================================================

class TSDataset(_BaseTrajectoryDataset):
    """
    Dataset for raw time-series arrays that need trajectory partitioning.

    Input is a single long sequence. The sequence is split into overlapping
    windows using sliding windows:

    - context length: `context_len` (conditioning timesteps)
    - prediction length: `pred_len` (forecast timesteps)
    - total window length: `context_len + pred_len`
    - sampling stride across windows: `stride`

    Windows whose context segment is entirely unobserved (all-zero mask across
    all context timesteps/features) are dropped.

    Resulting samples follow the base output schema with `x_*` and optional
    `y_*` fields.
    """

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        context_len: int,
        pred_len: int = 0,
        stride: int = 1,
        fillna: float | str | None = UNK,
        relative_timestamps: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data: Unsegmented time series (`pd.DataFrame` or `np.ndarray`).
            context_len: Number of context timesteps per sample.
            pred_len: Number of prediction timesteps per sample.
                If 0, dataset returns reconstruction-only samples (`x_*` only).
            stride: Step size between consecutive windows.
            fillna: Missing-value handling forwarded to `DataPack`.
            relative_timestamps: If True, each sample's `x_tps` starts at 0.
            verbose: If True, print sample count summary.
        """
        super().__init__(pred_len=pred_len, relative_timestamps=relative_timestamps)
        self.context_len = int(context_len)
        self.stride = int(stride)
        self.verbose = bool(verbose)

        pack = DataPack.from_time_series(data=data, fillna=fillna)
        x, m, t = pack.data, pack.mask, pack.ts  # [total_len, dim_x], [total_len, dim_x], [total_len]
        total = self.context_len + self.pred_len
        if x.shape[0] < total:
            raise ValueError(f"sequence too short: L={x.shape[0]}, need >= {total}")

        # Create sliding windows of length (context_len + pred_len)
        x_win = sliding_window_view(x, total, axis=0)  # [n_windows_raw, dim_x, window_size]
        m_win = sliding_window_view(m, total, axis=0)  # [n_windows_raw, dim_x, window_size]
        t_win = sliding_window_view(t, total, axis=0)  # [n_windows_raw, window_size]
        x_win = np.moveaxis(x_win, -1, 1)[:: self.stride]  # [n_windows, window_size, dim_x]
        m_win = np.moveaxis(m_win, -1, 1)[:: self.stride]  # [n_windows, window_size, dim_x]
        t_win = t_win[:: self.stride]                       # [n_windows, window_size]

        # Filter out windows with entirely unobserved context
        valid = m_win[:, : self.context_len].any(axis=tuple(range(1, m_win.ndim)))
        x_win = x_win[valid]  # [n_valid, window_size, dim_x]
        m_win = m_win[valid]  # [n_valid, window_size, dim_x]
        t_win = t_win[valid]  # [n_valid, window_size]

        for i in range(x_win.shape[0]):
            x_ctx = x_win[i, : self.context_len]  # [context_len, dim_x]
            m_ctx = m_win[i, : self.context_len]  # [context_len, dim_x]
            t_ctx = t_win[i, : self.context_len]  # [context_len]
            if self.pred_len == 0:
                self._append_sample(x=x_ctx, x_mask=m_ctx, x_tps=t_ctx)
            else:
                x_tgt = x_win[i, self.context_len :]  # [pred_len, dim_x]
                m_tgt = m_win[i, self.context_len :]  # [pred_len, dim_x]
                t_tgt = t_win[i, self.context_len :]  # [pred_len]
                self._append_sample(x=x_ctx, x_mask=m_ctx, x_tps=t_ctx, y=x_tgt, y_mask=m_tgt, y_tps=t_tgt)

        if self.verbose:
            print(f"[TSDataset] samples={len(self)} context_len={self.context_len} pred_len={self.pred_len}")


class TrajDataset(_BaseTrajectoryDataset):
    """
    Dataset for already segmented trajectories.

    Each input item is either:
    - dict: `{"traj": ..., "tps": ..., "mask": ...}` (`mask` optional), or
    - tuple: `(traj, tps, mask)` where `mask` may be `None`.

    Splitting behavior:
    - if `context_len` is provided: context is first `context_len` steps;
    - else if `pred_len > 0`: context is first `L_traj - pred_len` steps;
    - else: full trajectory is context-only (`x_*` only).

    Returned samples match `TSDataset`'s output contract exactly.
    """

    def __init__(
        self,
        data: Sequence[tuple | dict],
        pred_len: int = 0,
        context_len: int | None = None,
        fillna: float | str | None = UNK,
        relative_timestamps: bool = True,
    ):
        """
        Args:
            data: Sequence of trajectory samples (dicts or tuples).
            pred_len: Prediction horizon length. If 0, return context-only samples.
            context_len: Optional explicit context length override.
            fillna: Missing-value handling forwarded to `DataPack`.
            relative_timestamps: If True, shift timestamps so context starts at 0.
        """
        super().__init__(pred_len=pred_len, relative_timestamps=relative_timestamps)
        self.context_len = context_len

        for item in data:
            if isinstance(item, dict):
                traj = item["traj"]
                tps = item["tps"]
                mask = item.get("mask")
            elif isinstance(item, tuple) and len(item) == 3:
                traj, tps, mask = item
            else:
                raise TypeError("Each item must be dict(traj,tps,mask?) or tuple(traj,tps,mask).")

            pack = DataPack.from_trajectory(traj=traj, tps=tps, mask=mask, fillna=fillna)
            x, m, t = pack.data, pack.mask, pack.ts  # [traj_len, dim_x], [traj_len, dim_x], [traj_len]

            t_len = x.shape[0]
            if self.context_len is not None:
                cut = int(self.context_len)
            elif self.pred_len > 0:
                cut = t_len - self.pred_len
            else:
                cut = t_len

            if cut <= 0 or cut > t_len:
                raise ValueError(f"Invalid split point cut={cut} for trajectory length={t_len}.")
            if self.pred_len > 0 and cut + self.pred_len > t_len:
                raise ValueError(
                    f"context_len + pred_len exceeds trajectory length: {cut}+{self.pred_len}>{t_len}"
                )

            # Split trajectory into context and prediction segments
            x_ctx, m_ctx, t_ctx = x[:cut], m[:cut], t[:cut]  # [cut, dim_x], [cut, dim_x], [cut]
            if self.pred_len == 0:
                self._append_sample(x=x_ctx, x_mask=m_ctx, x_tps=t_ctx)
            else:
                x_tgt = x[cut : cut + self.pred_len]  # [pred_len, dim_x]
                m_tgt = m[cut : cut + self.pred_len]  # [pred_len, dim_x]
                t_tgt = t[cut : cut + self.pred_len]  # [pred_len]
                self._append_sample(x=x_ctx, x_mask=m_ctx, x_tps=t_ctx, y=x_tgt, y_mask=m_tgt, y_tps=t_tgt)
