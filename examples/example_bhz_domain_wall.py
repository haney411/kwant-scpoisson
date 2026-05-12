"""BHZ + magnetic exchange with an Ising domain wall along x.

Demonstrates: build the QAH-regime BHZ system with a tanh domain wall in m_z,
diagonalize the closed Nx × Ny system, and look at the low-|E| spectrum.

In the QAH regime, the closed system has chiral modes running around the
perimeter of each Chern-phase region. The domain wall itself also hosts a
1D chiral mode (the wall's "kink" mode). This example finds the in-gap
states and picks the one with the largest weight at the wall.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scpoisson import bhz


def main():
    Nx, Ny = 60, 24
    A, B, C, D, M = 1.0, 1.0, 0.0, 0.0, 0.5
    G_E, G_H = 2.0, 0.0
    m0 = 1.0
    x0 = Nx / 2.0

    m_func = bhz.domain_wall_magnetization(
        axis="x", center=x0, width=2.0, m0=m0, kind="ising",
    )

    syst = bhz.make_bhz_system(
        Nx=Nx, Ny=Ny,
        A=A, B=B, C=C, D=D, M=M,
        G_E=G_E, G_H=G_H,
        magnetization=m_func,
        leads=False,
    )
    H = syst.hamiltonian_submatrix()
    if hasattr(H, "toarray"):
        H = H.toarray()

    evals, evecs = np.linalg.eigh(H)
    print(f"H shape       : {H.shape}")
    print(f"Spectrum span : [{evals.min():+.3f}, {evals.max():+.3f}]")

    # Show eigenvalues near zero (the QAH in-gap modes).
    near0 = np.argsort(np.abs(evals))[:20]
    print("\nLowest-|E| eigenvalues (in-gap candidates):")
    for i in near0[:10]:
        print(f"  E = {evals[i]:+.5f}")

    # For each of those, find the wavefunction weight in a window around the
    # wall (|x - x0| < 4). The state with the largest such weight is our best
    # candidate for the wall-localized chiral mode.
    def wall_weight(psi):
        prob = (np.abs(psi.reshape(Nx, Ny, 4)) ** 2).sum(axis=(1, 2))
        return prob[max(0, int(x0 - 4)): int(x0 + 4)].sum()

    best = max(near0, key=lambda i: wall_weight(evecs[:, i]))
    psi = evecs[:, best]
    prob_x = (np.abs(psi.reshape(Nx, Ny, 4)) ** 2).sum(axis=(1, 2))

    print(f"\nWall-localized candidate: E = {evals[best]:+.5f}")
    print(f"  weight in |x-x0|<4 window: {wall_weight(psi):.3f}")
    print(f"  x-marginal (peak near x0 = {x0:.0f}):")
    pmax = prob_x.max()
    for x in range(Nx):
        bar = "#" * int(prob_x[x] / pmax * 40)
        marker = "  <-- wall" if x == int(x0) else ""
        print(f"  x={x:3d}  {bar}{marker}")


if __name__ == "__main__":
    main()
