# A Unified Neural State Space Model Codebase

## Overview

This repository provides a unified research codebase for deep-learning SSMs, facilitating modular experimentation across **Latent Discrete SSM**, **Latent Neural ODE** and **Latent Neural SDE**.

The framework is designed to be **lightweight**, implementing only the vanilla backbones of each model family without the overhead of extraneous variants. By utilizing a shared PyTorch Lightning training flow, common torch dataset construction & ingestion, and standardized forward-pass logic, the codebase enforces strict procedural invariance. This ensures that transitions between model families are seamless and comparisons are isolated to the core model formulations.

This repository serves as a functional complement to the review paper [Deep Learning-based Approaches for SSMs: A Selective Review (arXiv 2412.11211)](https://arxiv.org/pdf/2412.11211). While the codebase facilitates implementation, the paper provides a rigorous theoretical synthesis of the underlying formulations and the mathematical connections between these model families.

## Installation
This repo uses a local Python `3.11` virtual environment in `.venv`.

> Prerequisite: install `uv` and make sure it is available on your `PATH`. On Linux, one simple option is the official installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
uv --version
```

From the repo root:
```bash
bash setup.sh
```
`setup.sh` will:
- create a Python 3.11 `.venv` with `uv` 
- sync dependencies from `pyproject.toml`
- install the Git `pre-commit` hook
- register the Jupyter kernel as `neural-ssm`

## Repo Structure
```text
unified-neural-ssm/
├── src/
│   ├── models/         # Core model implementations, configs, and Lightning wrappers
│   ├── data/           # Torch dataset construction, trajectory partitioning, and batch collation
│   ├── losses/         # Reconstruction and KL/objective code used by the model trainers
│   └── utils/          # Shared training/runtime helpers
├── demo/
│   ├── README.md       # Practitioner guide for the demo workflow
│   ├── demo.ipynb      # Main end-to-end walkthrough notebook
│   └── lorenz96.py     # Demo-only synthetic data generator
├── docs/
│   ├── README.md       # Formulation and implementation notes
│   └── DIFFERENCES.md  # Differences from original upstream implementations
├── pyproject.toml      # Project metadata, dependencies, and tooling config
├── setup.sh            # Creates the local .venv and registers the Jupyter kernel
└── README.md           # Top-level guide
```

## For Practitioners
For most users, the practical workflow is:

- build samples with `TSDataset` or `TrajDataset`
- create `DataLoader` objects and use `prepare_batched_traj` as the collate function when you need aligned batches
- train `DiscreteSSM`, `ODESSM`, or `SDESSM`
- inspect fitted latent trajectories with `smooth` and generate forecasts with `forecast_from_context`

Start with:

- [demo/README.md](./demo/README.md) for the detailed workflow and forecasting API
- [demo/demo.ipynb](./demo/demo.ipynb) for the end-to-end synthetic example
- [docs/README.md](./docs/README.md) for formulation and implementation notes

## References
This codebase heavily references and adapts ideas from the following original repositories:

- Latent ODE (original): [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode)
- Latent SDE (original): [alluly/ident-latent-sde](https://github.com/alluly/ident-latent-sde)

More implementation details and differences from the referenced codebases are documented in [docs/README.md](./docs/README.md) and [docs/DIFFERENCES.md](./docs/DIFFERENCES.md).

## Citation
If you find this codebase useful, please cite as:

```text
@article{lin2024deep,
  title={Deep learning-based approaches for state space models: A selective review},
  author={Lin, Jiahe and Michailidis, George},
  journal={arXiv preprint arXiv:2412.11211},
  year={2024},
  url = {https://arxiv.org/pdf/2412.11211}
}
```
