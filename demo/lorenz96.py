from __future__ import annotations

import numpy as np


class SimLorenz:
    """
    State-space Lorenz-96 simulator with configurable latent and observation processes.

    Latent transition (Euler-Maruyama):
      z_{k+1} = z_k + f_l96(z_k) * dt + sigma(t_k) * sqrt(dt) * eps_k

    Observation model:
      x_k = W z_k + b + eta_k
    """

    def __init__(
        self,
        latent_dim: int = 5,
        obs_dim: int = 30,
        force: float = 2.8,
        delta_t: float = 0.05,
        burn_in: int = 1000,
        x0_scale: float = 0.01,
        sigma0: float = 0.10,
        sigma_rho: float = 0.15,
        sigma_omega: float = 0.15,
        sigma_phase: np.ndarray | None = None,
        sigma_min: float = 0.06,
        sigma_max: float = 0.14,
        obs_noise_scale: float = 0.05,
        emission_scale: float = 1.0,
        emission_bias: np.ndarray | None = None,
        emission_matrix: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if latent_dim < 3:
            raise ValueError(f"latent_dim must be >= 3 for Lorenz-96 coupling, got {latent_dim}.")
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}.")
        if delta_t <= 0:
            raise ValueError(f"delta_t must be > 0, got {delta_t}.")
        if burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {burn_in}.")
        if sigma_min <= 0 or sigma_max <= 0 or sigma_min > sigma_max:
            raise ValueError(
                f"Require 0 < sigma_min <= sigma_max, got sigma_min={sigma_min}, sigma_max={sigma_max}."
            )

        self.latent_dim = int(latent_dim)
        self.obs_dim = int(obs_dim)
        self.force = float(force)
        self.delta_t = float(delta_t)
        self.burn_in = int(burn_in)
        self.x0_scale = float(x0_scale)

        self.sigma0 = float(sigma0)
        self.sigma_rho = float(sigma_rho)
        self.sigma_omega = float(sigma_omega)
        if sigma_phase is None:
            self.sigma_phase = np.linspace(0.0, 2.0 * np.pi, self.latent_dim, endpoint=False, dtype=np.float64)
        else:
            sigma_phase_arr = np.asarray(sigma_phase, dtype=np.float64).reshape(-1)
            if sigma_phase_arr.shape[0] != self.latent_dim:
                raise ValueError(
                    f"sigma_phase length must match latent_dim={self.latent_dim}, got {sigma_phase_arr.shape[0]}."
                )
            self.sigma_phase = sigma_phase_arr
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.obs_noise_scale = float(obs_noise_scale)
        self.emission_scale = float(emission_scale)
        if emission_bias is None:
            self.emission_bias = np.zeros(self.obs_dim, dtype=np.float64)
        else:
            emission_bias_arr = np.asarray(emission_bias, dtype=np.float64).reshape(-1)
            if emission_bias_arr.shape[0] != self.obs_dim:
                raise ValueError(
                    f"emission_bias length must match obs_dim={self.obs_dim}, got {emission_bias_arr.shape[0]}."
                )
            self.emission_bias = emission_bias_arr
        if emission_matrix is None:
            self.emission_matrix = None
        else:
            emission_matrix_arr = np.asarray(emission_matrix, dtype=np.float64)
            if emission_matrix_arr.shape != (self.obs_dim, self.latent_dim):
                raise ValueError(
                    f"emission_matrix must have shape {(self.obs_dim, self.latent_dim)}, got {emission_matrix_arr.shape}."
                )
            self.emission_matrix = emission_matrix_arr

        self.seed = seed
        self.set_gc_matrix_()

    def set_gc_matrix_(self) -> None:
        """Set Lorenz-96 local coupling adjacency as Granger-causality ground truth."""
        p = self.latent_dim
        gc_matrix = np.zeros((p, p), dtype=int)
        for i in range(p):
            gc_matrix[i, i] = 1
            gc_matrix[i, (i + 1) % p] = 1
            gc_matrix[i, (i - 1) % p] = 1
            gc_matrix[i, (i - 2) % p] = 1
        self.gc_matrix = gc_matrix

    @staticmethod
    def _lorenz96_drift(z: np.ndarray, force: float) -> np.ndarray:
        """Lorenz-96 drift term f(z) for one latent state vector."""
        z_ip1 = np.roll(z, -1)
        z_im1 = np.roll(z, 1)
        z_im2 = np.roll(z, 2)
        return (z_ip1 - z_im2) * z_im1 - z + force

    def _sigma_t(self, t: float) -> np.ndarray:
        """Time-varying diffusion scale (state-independent) per latent coordinate."""
        sigma = self.sigma0 * (1.0 + self.sigma_rho * np.sin(self.sigma_omega * t + self.sigma_phase))
        return np.clip(sigma, self.sigma_min, self.sigma_max)

    def _get_rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(self.seed if seed is None else seed)

    def _build_emission_matrix(self, rng: np.random.Generator) -> np.ndarray:
        if self.emission_matrix is None:
            w = rng.normal(
                loc=0.0,
                scale=self.emission_scale,
                size=(self.obs_dim, self.latent_dim),
            )
            row_norm = np.linalg.norm(w, axis=1, keepdims=True).clip(min=1.0e-8)
            self.emission_matrix = w / row_norm
        return self.emission_matrix

    def sim_latent_process(
        self,
        n_steps: int,
        seed: int | None = None,
        burn_in: int | None = None,
        return_sigma: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a Lorenz-96 latent trajectory with Euler-Maruyama updates.

        Returns:
          latent: [n_steps, latent_dim]
          tps: [n_steps]
          sigma_path (optional): [n_steps, latent_dim]
        """
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}.")
        burn = self.burn_in if burn_in is None else int(burn_in)
        if burn < 0:
            raise ValueError(f"burn_in must be >= 0, got {burn}.")

        rng = self._get_rng(seed)
        total_steps = n_steps + burn
        sqrt_dt = np.sqrt(self.delta_t)

        latent_full = np.zeros((total_steps, self.latent_dim), dtype=np.float64)
        latent_full[0] = rng.normal(loc=0.0, scale=self.x0_scale, size=self.latent_dim)

        sigma_full = np.zeros((total_steps, self.latent_dim), dtype=np.float64)
        sigma_full[0] = self._sigma_t(0.0)

        for k in range(1, total_steps):
            t_prev = (k - 1) * self.delta_t
            z_prev = latent_full[k - 1]
            drift = self._lorenz96_drift(z_prev, self.force)
            sigma_t = self._sigma_t(t_prev)
            sigma_full[k] = sigma_t
            noise = rng.normal(loc=0.0, scale=1.0, size=self.latent_dim)
            latent_full[k] = z_prev + drift * self.delta_t + sigma_t * sqrt_dt * noise

        latent = latent_full[burn:]
        tps = np.arange(n_steps, dtype=np.float64) * self.delta_t
        sigma_path = sigma_full[burn:]

        if return_sigma:
            return latent.astype(np.float32), tps.astype(np.float32), sigma_path.astype(np.float32)
        return latent.astype(np.float32), tps.astype(np.float32)

    def sim_observation_process(
        self,
        latent: np.ndarray,
        seed: int | None = None,
        emission_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map latent states to observations via linear emission and Gaussian noise."""
        latent_arr = np.asarray(latent, dtype=np.float64)
        if latent_arr.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent last dim must be latent_dim={self.latent_dim}, got {latent_arr.shape[-1]}."
            )

        if emission_matrix is None:
            emission_rng = self._get_rng(seed)
            w = self._build_emission_matrix(emission_rng)
        else:
            w = np.asarray(emission_matrix, dtype=np.float64)
            if w.shape != (self.obs_dim, self.latent_dim):
                raise ValueError(
                    f"emission_matrix must have shape {(self.obs_dim, self.latent_dim)}, got {w.shape}."
                )

        obs_mean = np.einsum("...d,od->...o", latent_arr, w) + self.emission_bias.reshape(
            *((1,) * (latent_arr.ndim - 1)), self.obs_dim
        )
        if self.obs_noise_scale > 0:
            noise_rng = self._get_rng(seed)
            obs_noise = noise_rng.normal(loc=0.0, scale=self.obs_noise_scale, size=obs_mean.shape)
            obs = obs_mean + obs_noise
        else:
            obs = obs_mean
        return obs.astype(np.float32)

    def sim_long_series_state_space(
        self,
        n_steps: int,
        seed: int | None = None,
        return_emission_matrix: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate one long latent path and its observation path."""
        latent, tps = self.sim_latent_process(n_steps=n_steps, seed=seed, burn_in=self.burn_in)
        obs = self.sim_observation_process(latent, seed=seed)

        if return_emission_matrix:
            return obs, latent, tps, self.emission_matrix.astype(np.float32)
        return obs, latent, tps
