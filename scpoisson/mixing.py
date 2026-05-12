"""Mixing schemes for self-consistent iteration.

The SC loop computes an output potential V_out from an input V_in via
the density-Poisson sequence. Pure substitution (V_in <- V_out) often
oscillates; mixing schemes blend old iterates to stabilize.
"""

from __future__ import annotations

import numpy as np


class LinearMixer:
    """Simple linear mixing: V_new = (1 - α) V_in + α V_out."""

    def __init__(self, alpha=0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def step(self, V_in, V_out):
        return (1 - self.alpha) * V_in + self.alpha * V_out

    def reset(self):
        pass


class AndersonMixer:
    """Anderson (a.k.a. Pulay) mixing.

    Maintains a history of (V_in, residual=V_out-V_in) pairs and at each
    step solves a least-squares problem to find the linear combination
    that minimizes the predicted residual.

    Parameters
    ----------
    alpha : float
        Linear-mixing fraction in the trivial update.
    history : int
        Maximum number of past iterations to keep.
    """

    def __init__(self, alpha=0.3, history=5):
        self.alpha = alpha
        self.history = history
        self.reset()

    def reset(self):
        self._V = []   # past inputs
        self._F = []   # past residuals F = V_out - V_in

    def step(self, V_in, V_out):
        V_in = np.asarray(V_in, dtype=float)
        F = np.asarray(V_out, dtype=float) - V_in
        # Append current iterate.
        self._V.append(V_in.copy())
        self._F.append(F.copy())
        # Trim
        while len(self._V) > self.history + 1:
            self._V.pop(0)
            self._F.pop(0)

        m = len(self._F) - 1
        if m == 0:
            return V_in + self.alpha * F  # plain linear mixing on first step

        # Build DF matrix of shape (n_points, m): columns = F_k - F_{k-1}
        DF = np.stack([self._F[i + 1] - self._F[i] for i in range(m)], axis=1)
        DV = np.stack([self._V[i + 1] - self._V[i] for i in range(m)], axis=1)
        # Solve least squares  DF gamma = F_current
        gamma, *_ = np.linalg.lstsq(DF, F, rcond=None)
        V_new = V_in + self.alpha * F - (DV + self.alpha * DF) @ gamma
        return V_new
