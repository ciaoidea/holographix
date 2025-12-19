from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MindDynamics:
    dim: int
    dt: float = 0.05
    torsion_lr: float = 0.01
    torsion_decay: float = 0.001
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.rng = np.random.default_rng(self.seed)
        self.x = np.zeros(int(self.dim), dtype=np.float64)
        self.K = np.zeros((int(self.dim), int(self.dim)), dtype=np.float64)
        self._prev_u: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.x[...] = 0.0
        self.K[...] = 0.0
        self._prev_u = None

    def energy(self, u: np.ndarray) -> float:
        u = self._as_vec(u)
        d = self.x - u
        return 0.5 * float(np.dot(d, d))

    def step(self, u: np.ndarray, *, learn_torsion: bool = True) -> np.ndarray:
        u = self._as_vec(u)

        grad = self.x - u
        dx_pot = -(grad)
        dx_rot = self.K @ self.x

        self.x = self.x + float(self.dt) * (dx_pot + dx_rot)

        if learn_torsion:
            self._update_torsion(u)

        return self.x

    def _update_torsion(self, u: np.ndarray) -> None:
        if self._prev_u is None:
            self._prev_u = u.copy()
            return

        a = self._prev_u
        b = u
        dK = np.outer(b, a) - np.outer(a, b)
        self.K = (1.0 - float(self.torsion_decay)) * self.K + float(self.torsion_lr) * dK
        self.K = 0.5 * (self.K - self.K.T)
        self._prev_u = u.copy()

    def _as_vec(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.size != self.dim:
            raise ValueError(f"expected u to have shape ({self.dim},), got {u.shape}")
        return u
