import torch
import torch.nn as nn

# Minimum scale for standard deviation to prevent numerical instability
STD_MIN_SCALE = 1.0e-4


class MLP(nn.Module):
    """
    Feed-forward network:
      x -> Linear -> GELU -> ... -> Linear.

    Used as the standard parameterization block for drift/diffusion/decoder maps.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | list[int] | tuple[int, ...] | None,
        dropout: float = 0.0,
        activation: str = "GELU",
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = []
        elif isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        elif isinstance(hidden_dim, tuple):
            hidden_dim = list(hidden_dim)
        elif not isinstance(hidden_dim, list):
            raise TypeError(
                f"hidden_dim must be one of int, list, tuple, or None; got {type(hidden_dim)}"
            )

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            match activation:
                case "GELU":
                    layers.append(nn.GELU())
                case "ReLU":
                    layers.append(nn.ReLU())
                case "Tanh":
                    layers.append(nn.Tanh())
                case _:
                    raise ValueError(
                        f"Unsupported activation={activation}. "
                        "Expected one of {'GELU', 'ReLU', 'Tanh'}."
                    )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.ffnet = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape `[..., input_dim]`.

        Returns:
            Tensor of shape `[..., output_dim]` containing transformed features.
        """
        return self.ffnet(x)


class BiRNNEncoder(nn.Module):
    """
    Shared bidirectional sequence encoder for discrete-family posterior parameterization.

    Maps observations x_{1:T} to per-step posterior parameters q(z_t | x_{1:T})
    using a projected bidirectional GRU:
      x_t -> h_t -> (mu_t, sigma_t).
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        x_proj_dim: int,
        rnn_hidden_dim: int = 64,
        rnn_num_layers: int = 1,
        mask_processor_kernel_size: int = 3,
        dropout: float = 0.1,
        sigma_type: str = "const",
    ) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.sigma_type = sigma_type
        self.softplus = nn.Softplus()
        if self.sigma_type not in {"const", "diag"}:
            raise ValueError(
                f"Unsupported sigma_type={self.sigma_type}. Expected one of {{'const','diag'}}."
            )

        self.x_projection = MLP(
            input_dim=dim_x,
            output_dim=x_proj_dim,
            hidden_dim=[x_proj_dim],
            dropout=dropout,
            batch_norm=False,
        )

        self.mask_1dcnn = nn.Conv1d(dim_x, 2 * x_proj_dim, mask_processor_kernel_size, padding="same")
        self.mask_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * x_proj_dim, x_proj_dim),
        )

        self.birnn = nn.GRU(
            input_size=x_proj_dim + 1,
            hidden_size=self.rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.combiner_forward_mu = nn.Linear(rnn_hidden_dim, self.dim_z)
        self.combiner_backward_mu = nn.Linear(rnn_hidden_dim, self.dim_z)
        if self.sigma_type == "const":
            self.sigma_const_raw = nn.Parameter(torch.randn(1))
        else:
            self.combiner_forward_sigma = nn.Sequential(
                nn.Linear(rnn_hidden_dim, self.dim_z),
                nn.Softplus(),
            )
            self.combiner_backward_sigma = nn.Sequential(
                nn.Linear(rnn_hidden_dim, self.dim_z),
                nn.Softplus(),
            )

    @staticmethod
    def merge(mu1, sigma2_1, mu2, sigma2_2):
        mu = (mu2 * sigma2_1 + mu1 * sigma2_2) / (sigma2_1 + sigma2_2)
        sigma2 = sigma2_2 * sigma2_1 / (sigma2_1 + sigma2_2)
        return mu, sigma2

    @staticmethod
    def get_delta_t(tps):
        delta_t = tps[:, 1:] - tps[:, :-1]
        delta_t = torch.cat([torch.zeros(tps.shape[0], 1, device=delta_t.device), delta_t], dim=1)
        return delta_t.unsqueeze(-1)

    def _project_x(self, x):
        return self.x_projection(x)

    def _build_h0(self, batch_size, device, h0_forward=None, h0_backward=None):
        if h0_forward is None:
            h0_forward = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        if h0_backward is None:
            h0_backward = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        return torch.cat([h0_forward, h0_backward], dim=0)

    def forward(self, x, mask, tps, h0_forward=None, h0_backward=None):
        if x.shape[-1] != self.dim_x:
            raise ValueError(f"x.shape={x.shape}; expecting last dim {self.dim_x}")
        if x.shape != mask.shape:
            raise ValueError(f"x.shape={x.shape}; mask.shape={mask.shape}; expecting same shape")
        h0 = self._build_h0(x.shape[0], x.device, h0_forward, h0_backward)

        hx = self._project_x(x)
        mask_hidden = self.mask_projector(self.mask_1dcnn(mask.permute(0, 2, 1)).permute(0, 2, 1))

        hx = hx + mask_hidden
        delta_t = self.get_delta_t(tps)
        hx = torch.cat([hx, delta_t], dim=-1)

        h, _ = self.birnn(hx, h0)
        h_l, h_r = h[..., : self.rnn_hidden_dim], h[..., self.rnn_hidden_dim :]

        mu_l = self.combiner_forward_mu(h_l)
        mu_r = self.combiner_backward_mu(h_r)
        if self.sigma_type == "const":
            mu = 0.5 * (mu_l + mu_r)
            sigma_std_scalar = self.softplus(self.sigma_const_raw).to(mu) + STD_MIN_SCALE
            sigma2 = (sigma_std_scalar**2).view(1, 1, 1).expand_as(mu)
        else:
            sigma_l = self.combiner_forward_sigma(h_l) + STD_MIN_SCALE
            sigma_r = self.combiner_backward_sigma(h_r) + STD_MIN_SCALE
            sigma2_l = sigma_l**2
            sigma2_r = sigma_r**2
            mu, sigma2 = self.merge(mu_l, sigma2_l, mu_r, sigma2_r)
        return mu, sigma2, h_l, h_r


class RNNEncoder(nn.Module):
    """
    Shared sequence encoder for ODE/SDE posterior parameterization.

    Per-step encoder input:
      [x_t, m_t, delta_t]

    Direction handling matches the ODE encoder behavior:
      - reverse sequence inputs when `encoding_direction='backward'`
      - reverse `delta_t` accordingly
      - run one-direction GRU
      - reverse outputs back to original time order

    Sigma parameterization:
      - `const`: single global scalar std, shape []
      - `diag`: per-step/per-dim std, shape [B, L, dim_z]
      - `full`: packed lower-triangle factor, shape [B, L, dim_z*(dim_z+1)//2]
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        proj_dim: int,
        sigma_type: str = "diag",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if sigma_type not in ("const", "diag", "full"):
            raise ValueError(
                f"Unsupported sigma_type={sigma_type}. "
                "Expected one of {'const', 'diag', 'full'}."
            )

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sigma_type = sigma_type

        input_dim = 2 * dim_x + 1
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(rnn_hidden_dim, proj_dim),
        )

        self.mu_head = nn.Linear(proj_dim, dim_z)
        if sigma_type == "const":
            self.sigma_const_raw = nn.Parameter(torch.randn(1))
        elif sigma_type == "diag":
            self.sigma_head = nn.Sequential(
                nn.Linear(proj_dim, dim_z),
                nn.Softplus(),
            )
        else:
            tri_dim = dim_z * (dim_z + 1) // 2
            self.sigma_head = nn.Linear(proj_dim, tri_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        tps: torch.Tensor,
        encoding_direction: str = "forward",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode posterior parameters on a batched time grid.

        Args:
          x: [B, L, dim_x], observed values.
          mask: [B, L, dim_x], observation mask.
          tps: [B, L], timestamps aligned with `x`.
          encoding_direction: "forward" or "backward".

        Returns:
          q_mu: [B, L, dim_z], posterior mean path.
          q_sigma:
            - [B, L, dim_z] for `diag`,
            - [] for `const`,
            - [B, L, dim_z*(dim_z+1)//2] for `full`.
        """
        if encoding_direction not in {"forward", "backward"}:
            raise ValueError(
                f"Unsupported encoding_direction={encoding_direction}. "
                "Expected one of {'forward', 'backward'}."
            )

        data = torch.cat([x, mask], dim=-1)
        delta_t = tps[:, 1:] - tps[:, :-1]

        if encoding_direction == "backward":
            data = torch.flip(data, dims=[1])
            delta_t = torch.flip(delta_t, dims=[1])
            
        delta_t = torch.cat(
            [delta_t, torch.zeros(tps.shape[0], 1, device=tps.device, dtype=tps.dtype)],
            dim=1,
        ).unsqueeze(-1)
        data = torch.cat([data, delta_t], dim=-1)

        h, _ = self.gru(data)
        h = self.proj(h)
        
        if encoding_direction == "backward":
            h = torch.flip(h, dims=[1])

        q_mu = self.mu_head(h)
        if self.sigma_type == "diag":
            q_sigma = self.sigma_head(h) + STD_MIN_SCALE
        elif self.sigma_type == "const":
            q_sigma = torch.nn.functional.softplus(self.sigma_const_raw).reshape(()) + STD_MIN_SCALE
        else:
            q_sigma = self.sigma_head(h)
        return q_mu, q_sigma


class GaussianObsDecoder(nn.Module):
    """
    Shared Gaussian observation decoder used by Discrete/ODE/SDE models.

    Parameterization:
      p(x_t | z_t) = N(g(z_t), x_sigma2 * I)

    where `g` is an MLP and `x_sigma2` is a fixed scalar variance.
    """

    def __init__(
        self,
        dim_z: int,
        dim_x: int,
        hidden_dim: int | list[int] | tuple[int, ...] | None,
        x_sigma2: float,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mean_head = MLP(
            input_dim=dim_z,
            output_dim=dim_x,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.x_sigma2 = x_sigma2

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent states into Gaussian observation statistics.

        Args:
          z: [..., dim_z], latent states.

        Returns:
          x_mean: [..., dim_x], decoded observation means.
          x_sigma2: [..., dim_x], fixed observation variances (all entries = x_sigma2).
        """
        x_mean = self.mean_head(z)
        x_sigma2 = self.x_sigma2 * torch.ones_like(x_mean)
        return x_mean, x_sigma2



class Sampler:
    @staticmethod
    def sample_normal(mu, var, mc_samples=1, is_logvar=False, take_mc_average=True):
        """
        Reparameterized sampling from Normal(mu, var).

        z = mu + sigma * eps,  eps ~ N(0, I),  sigma = sqrt(var)

        If `is_logvar=True`, input `var` is interpreted as log-variance.
        If `take_mc_average=True`, returns the Monte Carlo mean across samples.

        Args:
            mu: Mean tensor with shape `[..., D]`.
            var: Variance tensor with shape `[..., D]` (or log-variance if
                `is_logvar=True`).
            mc_samples: Number of Monte Carlo samples.
            is_logvar: Whether `var` is log-variance.
            take_mc_average: If True, returns shape `[..., D]`; otherwise
                returns shape `[mc_samples, ..., D]`.
        Returns:
            Sample tensor from N(mu, var):
              - `[..., D]` if `take_mc_average=True` (MC mean sample),
              - `[mc_samples, ..., D]` otherwise (raw MC samples).
        """
        assert mu.shape == var.shape
        std = torch.exp(0.5 * var) if is_logvar else torch.sqrt(var)
        mc = mu + std * torch.randn((mc_samples, *mu.shape), device=mu.device)
        if not take_mc_average:
            return mc
        return mc.mean(axis=0, keepdims=False)

    @staticmethod
    def sample_normal_full(
        mu: torch.Tensor,
        cov: torch.Tensor,
        mc_samples: int = 1,
        take_mc_average: bool = True,
        is_cholesky: bool = False,
        eps: float = 1.0e-12,
    ) -> torch.Tensor:
        """
        Reparameterized sampling from a full-covariance Gaussian distribution.

        Sampling uses a Cholesky factorization:
          L = chol(Sigma + eps I),  z = m + L eps,  eps ~ N(0, I),
        or directly uses a provided Cholesky factor when `is_cholesky=True`.

        Args:
            mu: Mean tensor with shape `[... , D]`.
            cov: Covariance tensor with shape `[... , D, D]`, or a lower-triangular
                Cholesky factor with the same shape when `is_cholesky=True`.
            mc_samples: Number of Monte Carlo samples.
            take_mc_average: If True, returns shape `[... , D]`; otherwise
                returns shape `[mc_samples, ... , D]`.
            is_cholesky: If True, interpret `cov` as a lower-triangular Cholesky factor.
            eps: Diagonal jitter added before Cholesky factorization.
        Returns:
            Sample tensor from the target Gaussian:
              - `[... , D]` if `take_mc_average=True`,
              - `[mc_samples, ... , D]` otherwise.
        """
        if mu.shape[:-1] != cov.shape[:-2]:
            raise ValueError(f"mu leading shape {mu.shape[:-1]} must match cov leading shape {cov.shape[:-2]}")
        if cov.shape[-1] != cov.shape[-2] or cov.shape[-1] != mu.shape[-1]:
            raise ValueError("Expected cov shape [..., D, D] matching mu shape [..., D].")

        d = mu.shape[-1]
        if is_cholesky:
            chol = cov
        else:
            eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
            chol = torch.linalg.cholesky(cov + eps * eye)
        noise = torch.randn((mc_samples, *mu.shape, 1), device=mu.device, dtype=mu.dtype)  # [S, ..., D, 1]
        mc = mu.unsqueeze(0) + torch.matmul(chol.unsqueeze(0), noise).squeeze(-1)  # [S, ..., D]
        if not take_mc_average:
            return mc
        return mc.mean(dim=0)
