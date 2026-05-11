# Demo

This folder contains the demo notebook and the helper code used by that notebook. The reusable model and training code lives in `src/`.

## Files
- [demo.ipynb](./demo.ipynb): end-to-end experiment on synthetic trajectories
- `lorenz96.py`: Lorenz-96 simulator used to generate the demo data

## End-To-End Workflow
The demo follows the same backbone for the supported model families.

1. Generate or load time-series data.
2. Build a dataset:
Use `TSDataset(...)` when you start from one long time series and want the repo to partition it into windows. Use `TrajDataset(...)` when you already have trajectory windows or trajectory segments.
3. Create `DataLoader` objects and use `prepare_batched_traj(...)` as the `collate_fn`.
4. Choose one model family and its matching config pair.
5. Train with `configure_trainer(...)` and `trainer.fit(...)`.
6. After training, use `smooth(...)` for latent inference on the observed context and `forecast_from_context(...)` for future prediction. Use `reconstruct(...)` only when you want in-sample fit diagnostics or visualization.

A minimal pattern looks like:

```python
# Option 1: start from one long time series
dataset = TSDataset(
    data=series,
    context_len=CONTEXT_LEN,
    pred_len=0,
    stride=STRIDE,
)

# Option 2: start from pre-windowed trajectory samples
traj_samples = [
    {"traj": x_window, "tps": t_window, "mask": m_window},
    ...
]
dataset = TrajDataset(data=traj_samples)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda batch: prepare_batched_traj(batch, cutoff_len=None),
)

model_cfg = SDEModelConfig(...)
opt_cfg = SDEOptConfig(...)
lightning_module = SDESSM(model_cfg, opt_cfg)
trainer = configure_trainer(opt_cfg, output_dir="output/experiment")
trainer.fit(lightning_module, train_dataloaders=loader)
```

If you want a split context/prediction setup, pass `prepare_batched_traj(batch, cutoff_len=CONTEXT_LEN)` as the `collate_fn` so each batch includes both `x_*` and `y_*`.

## Forecast API
After `trainer.fit(...)`, call forecasting on the trained Lightning module's `.model` attribute.

The common unbatched inputs are:

- `x`: observed context, shape `[L_obs, dim_obs]`
- `mask`: context mask, shape `[L_obs, dim_obs]`
- `time_steps`: context timestamps, shape `[L_obs]`
- `time_steps_to_predict`: forecast timestamps, shape `[L_pred]`

Across model families, `n_trajs` controls the forecast mode:

- `n_trajs=1`: deterministic forecast with no latent noise and no observation noise
- `n_trajs>1`: sampled forecast trajectories with model-specific stochasticity

Typical calls are:

```python
core_model = lightning_module.model

# DiscreteSSM or SDESSM: deterministic point forecast
x_pred, z_pred = core_model.forecast_from_context(
    x=x,
    mask=mask,
    time_steps=time_steps,
    time_steps_to_predict=time_steps_to_predict,
    n_trajs=1,
)

# DiscreteSSM or SDESSM: sampled forecast trajectories
x_samples, z_samples = core_model.forecast_from_context(
    x=x,
    mask=mask,
    time_steps=time_steps,
    time_steps_to_predict=time_steps_to_predict,
    n_trajs=32,
)

# ODESSM: same n_trajs convention, plus anchor_mode and returned forecast grid
x_pred, z_pred, t_pred = core_model.forecast_from_context(
    x=x,
    mask=mask,
    time_steps=time_steps,
    time_steps_to_predict=time_steps_to_predict,
    n_trajs=1,
    anchor_mode="last",
)
```

The main practical differences are:

- `DiscreteSSM` and `SDESSM` return `(x_pred, z_pred)`.
- `ODESSM` returns `(x_pred, z_pred, t_pred)` and uses `anchor_mode` to choose whether the rollout is anchored at the inferred initial state or the smoothed last latent state.
- For `ODESSM`, `n_trajs>1` samples multiple latent anchor states before solving the deterministic ODE.
- If you train `ODESSM` with `train_strategy='predict'`, training batches must include `y_*`.

If you need the model-internal formulation or implementation details, see [../docs/README.md](../docs/README.md).
