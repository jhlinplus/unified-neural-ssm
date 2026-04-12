# Differences from the Original Implementations

This document highlights the differences between the current implementations and their original codebases.

---

## 1. Latent Neural SDE

Differences from the original latent neural SDE [alluly/ident-latent-sde](https://github.com/alluly/ident-latent-sde) codebase.

### 1.1 Posterior Encoder Architecture and Time Awareness

**Component**: Inference model $q\_\phi(z\_t\mid \cdot)$ for SDE.

**Original implementation**:
Posterior parameters are produced with an MLP encoder on observed $x_t$ (framewise), without a temporal sequence model:

$$
(\mu\_t^q,\sigma\_t^q) = \mathrm{Encoder}\_\phi(x\_t).
$$

**Current implementation**:
Uses a sequence encoder with time-gap features:

$$
(\mu\_t^q,\sigma\_t^q) = \mathrm{Encoder}\_\phi(x\_{1:L},m\_{1:L},t\_{1:L}),
$$

where encoder input includes $(x\_t, m\_t, \Delta t)$ and a single-direction GRU
(forward or backward according to `enc_direction`, matching ODE-style handling).

**Impact**:
- Current posterior inference is explicitly time-aware under irregular spacing through $\Delta t$.
- This is intentionally different from the original framewise MLP inference design.
- Transition/KL math remains in the same latent SDE family; the difference is in posterior parameterization capacity.

### 1.2 Posterior Sampling (Different Reparameterization)

**Component**: Encoder sampling during training.

**Standard reparameterization**:
For $q = \mathcal N(\mu, \Sigma)$ with Cholesky factor $\Sigma = LL^\top$:

$$
z = \mu + L\varepsilon, \quad \varepsilon \sim \mathcal N(0,I).
$$

This produces samples with covariance $\mathbb{E}[(z-\mu)(z-\mu)^\top] = L\mathbb{E}[\varepsilon\varepsilon^\top]L^\top = LL^\top = \Sigma$.

**Original implementation**:
Samples as:

$$
z = \mu + (LL^\top)\varepsilon = \mu + \Sigma\varepsilon.
$$

This produces covariance $\mathbb{E}[\Sigma\varepsilon\varepsilon^\top\Sigma^\top] = \Sigma\Sigma = (LL^\top)^2 \neq \Sigma$.

**Current implementation**:
Uses $z = \mu + L\varepsilon$.

**Impact**: Stochastic latent paths sampled during training have different variance. Affects all KL modes when using full covariance posterior.

### 1.3 KL Divergence Computation

**Component**: Latent regularization term $\mathcal L\_{\mathrm{KL}}$.

#### 1.3.1 `exact` mode with Full Covariance (Materially Different)

**Standard formulation**:
For Gaussian increment distributions $q\_{\mathrm{inc}} = \mathcal N(\mu\_q, \Sigma\_q)$ and $p\_{\mathrm{inc}} = \mathcal N(\mu\_p, \Sigma\_p)$, the KL divergence is:

$$
\mathrm{KL}(q\_{\mathrm{inc}} \| p\_{\mathrm{inc}}) = \frac{1}{2}\left[\mathrm{tr}(\Sigma\_p^{-1}\Sigma\_q) + (\mu\_p - \mu\_q)^\top \Sigma\_p^{-1} (\mu\_p - \mu\_q) - D + \log\frac{|\Sigma\_p|}{|\Sigma\_q|}\right].
$$

**Original implementation**:
The `kl_div_exact` full-covariance branch uses a non-standard expression:
- Quadratic term: scales by $\Delta t$ instead of $(\Sigma\_p \Delta t)^{-1}$
- Trace term: uses diagonal-only contraction, not the full $\mathrm{tr}(\Sigma\_p^{-1}\Sigma\_q)$
- Log-determinant: computed differently

This yields a different objective and different gradients.

**Current implementation**:
Uses the standard formula with
$q\_{\mathrm{inc},t} = \mathcal N(q\_{\mu,t+1}-q\_{\mu,t}, \Sigma\_{t+1}(q\_\sigma)+\Sigma\_t(q\_\sigma))$ and
$p\_{\mathrm{inc},t} = \mathcal N(\mu\_t^{p,\mathrm{inc}}\Delta t, \Sigma\_t^\theta \Delta t)$.

**Impact**: Models trained with full covariance under `exact` mode produce different results due to different objective gradients.

#### 1.3.2 `exact` mode with Diagonal Covariance (Effectively Equivalent)

**Standard formulation**:
For diagonal covariances, the KL simplifies to:

$$
\mathrm{KL}(q \| p) = \frac{1}{2}\sum\_{d=1}^D\left[\frac{\sigma\_{q,d}^2}{\sigma\_{p,d}^2} + \frac{(\mu\_{p,d} - \mu\_{q,d})^2}{\sigma\_{p,d}^2} - 1 + \log\frac{\sigma\_{p,d}^2}{\sigma\_{q,d}^2}\right].
$$

**Original implementation**:
- Omits the $-D/2$ (or $-1$ per dimension) constant term
- Applies `abs()` to scale/variance terms

**Current implementation**:
Uses the standard diagonal KL formula.

**Impact**: The difference is an additive constant. Under standard positive parameterization, gradients are identical, so optimization behavior is the same.

### 1.4 Reconstruction Mode

**Component**: Observation reconstruction term $\mathcal L\_{\mathrm{rec}}$.

**Original implementation**:
Boolean flag `plus_one` toggled between two mutually exclusive reconstruction modes:

**`plus_one=False` (direct reconstruction)**: Decode current posterior samples to predict current observations

$$
\mathcal L\_{\mathrm{rec}} = \sum\_{t=1}^L \ell(x\_t, x\_t^{\mathrm{rec}}),\quad x\_t^{\mathrm{rec}} = g\_\theta^{(x)}(z\_t), \quad z\_t \sim q\_\phi(z\_t\mid x\_{1:L}).
$$

**`plus_one=True` (one-step-ahead reconstruction)**: Transition from posterior-mean anchors (teacher forcing) and predict next observations

$$
\mathcal L\_{\mathrm{rec}} = \sum\_{t=1}^{L-1} \ell(x\_{t+1}, x\_{t+1}^{+}),
$$

where $z\_{t+1}^{+} = z\_t + f\_\theta(t, q\_{\mu,t})\Delta t + g\_\theta(t, q\_{\mu,t})\Delta W\_t$ and $x\_{t+1}^{+} = g\_\theta^{(x)}(z\_{t+1}^{+})$.

The flag selected **one mode or the other**, not a sum of both.

**Current implementation**:
Only supports plus-one reconstruction (`plus_one=True` equivalent), also using posterior-mean anchors $q\_{\mu,t}$ for the transition step. Sampled path $z\_t$ is used only in the `mc` KL branch.

**Commonality**: Both implementations use teacher forcing (posterior-mean anchors) within the plus-one reconstruction mode.

**Difference**: Original supported both direct and plus-one modes; current only supports plus-one mode.

**Impact**: Different reconstruction objective. Plus-one mode emphasizes learning accurate latent dynamics, while direct mode emphasizes encoding-decoding fidelity. Training signal and learned representations differ between modes.

---

## 2. Latent Neural ODE

Differences from the original latent neural ode [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) codebase.

### 2.1 Inference Family for the Initial Latent State

**Component**: Posterior model $q\_\phi(z\_0 \mid \cdot)$.

**Original implementation**:
Supports two inference families for $q\_\phi(z\_0\mid x)$:

1. RNN encoder:

$$
q\_\phi(z\_0\mid x\_{1:L},m\_{1:L},t\_{1:L}) = \mathcal N(\mu\_0^q,\mathrm{diag}(\sigma\_0^{2,q})).
$$

2. ODE-RNN encoder:
   the recognition trajectory itself evolves continuously between observations and is updated at observation times before producing $(\mu\_0^q,\sigma\_0^q)$.

So the original inference model can be either purely discrete-time recurrent or continuous-time recurrent.

**Current implementation**:
Uses a single RNN-based inference family:

$$
q\_\phi(z\_0\mid x\_{1:L},m\_{1:L},t\_{1:L}) = \mathcal N(\mu\_0^q,\mathrm{diag}(\sigma\_0^{2,q})),
$$

with $\Delta t$ supplied as an encoder input feature.

**Impact**: The current model removes the ODE-RNN recognition family, so inference-time continuous latent evolution in the encoder is not part of the current ODE pipeline.

### 2.2 Training Objective: IWAE-Style vs ELBO-Style

**Component**: Variational training objective.

**Original implementation**:
Uses an IWAE-style aggregation across $S$ latent samples:

$$
\mathcal L\_{\text{orig}} = -\log \sum\_{s=1}^{S}\exp\,\left(\log p\_\theta(x\_{\text{target}}\mid z^{(s)}(\cdot))- \beta\_{\mathrm{KL}}\,\mathrm{KL}(q\_\phi(z\_0\mid x)\|p(z\_0))\right),
$$

with $z\_0^{(s)}\sim q\_\phi(z\_0\mid x)$ and $z^{(s)}(\cdot)$ from ODE solve.

**Current implementation**:
Supports both aggregation modes via `loss_mode`:

**`iwae`**:

$$
\mathcal L\_{\text{cur}}^{\text{iwae}} = -\log \sum\_{s=1}^{S} \exp,\left(\log p\_\theta(x\_{\text{target}}\mid z^{(s)}(\cdot)) - \beta\_{\mathrm{KL}}\,\mathrm{KL}(q\_\phi(z\_0\mid x)\|p(z\_0))\right).
$$

**`elbo`**:

$$
\mathcal L\_{\text{cur}}^{\text{elbo}} = \frac{1}{S}\sum\_{s=1}^{S}\mathcal L\_{\mathrm{rec}}^{(s)} + \beta\_{\mathrm{KL}}\,\mathrm{KL}(q\_\phi(z\_0\mid x)\|p(z\_0)).
$$

Current default is `loss_mode='iwae'`.

**Impact**: At the high level, both original and current ODE training remain reconstruction + KL objectives. The main difference is aggregation/normalization detail (for example, log-sum-exp normalization constants and reduction choices), not a change in the overall objective family.

### 2.3 Reconstruction Likelihood Scaling

**Component**: Observation reconstruction term.

**Original implementation**:
Uses masked Gaussian log-likelihood with explicit normalization by the number of observed entries (per attribute / trajectory aggregation), then combines across trajectories.

**Current implementation**:
Uses Gaussian reconstruction with either NLL $-\log \mathcal N(x\_{\text{target}}; \hat x, \sigma\_x^2 I)$ or MSE mode (controlled by `loss_type`), with entry-weighted global reduction (`mean` or `sum`) over selected entries. This is kept intentionally for consistency across ODE/Discrete/SDE trainers.

**Impact**: This is a weighting-detail difference, not an objective-family difference. In practice it is usually modest, and is most noticeable only when observation counts vary strongly across trajectories/features (highly uneven missingness).
