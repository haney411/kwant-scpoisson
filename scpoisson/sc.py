"""Top-level self-consistent Poisson loop.

Iterates:
  1. Compute electron density n_e(orb) from kwant under current V_onsite(orb).
  2. Solve Poisson for V_new(orb) given n_e and the user-supplied poisson
     callable (1D, 2D, custom).
  3. Mix V_new with V to get next iterate.
Until ||V_new - V||_∞ < tol.

The Poisson step is **user-supplied**: any callable that takes the flat
density-source array (``n_e - n_background``, shape (N_orb,)) and returns
a flat potential array of the same shape. Use the factory helpers
``poisson_solve_1d`` and ``poisson_solve_2d`` for the common cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np

from . import density, poisson, mixing


@dataclass
class SCResult:
    V: np.ndarray
    n: np.ndarray
    iterations: int
    converged: bool
    residual_history: List[float] = field(default_factory=list)


# --------------------------------------------------------------------- #
# Convenience factories for built-in Poisson solvers                     #
# --------------------------------------------------------------------- #

def poisson_solve_1d(
    *,
    dx,
    coupling=1.0,
    eps_r=1.0,
    bc_left=("dirichlet", 0.0),
    bc_right=("dirichlet", 0.0),
):
    """Return a callable src_flat -> V_flat for a 1D Poisson on N orbitals.

    The orbital index equals the longitudinal grid index (one orbital per
    site is assumed). For multi-orbital sites or quasi-2D systems use
    `poisson_solve_2d` or write your own callable.
    """
    def solve(src):
        return poisson.solve_1d_natural(
            src, dx=dx, eps_r=eps_r, coupling=coupling,
            bc_left=bc_left, bc_right=bc_right,
        )
    return solve


def poisson_solve_2d(
    *,
    grid_shape,                          # (Nx, Ny)
    orbital_grid,                        # (Nx, Ny) int array: orbital index of each cell
    dx,
    dy=None,
    coupling=1.0,
    eps_r=1.0,
    bc_x_left=("dirichlet", 0.0),
    bc_x_right=("dirichlet", 0.0),
    bc_y_bot=("neumann", 0.0),
    bc_y_top=("neumann", 0.0),
):
    """Return a callable src_flat -> V_flat for a 2D Poisson on (Nx, Ny).

    Parameters
    ----------
    grid_shape : (Nx, Ny)
    orbital_grid : (Nx, Ny) int array
        Map from (x, y) cell to a flat orbital index in the kwant system.
        Use `column_or_grid_map(syst, ...)` helpers to construct this.
    """
    Nx, Ny = grid_shape
    orbital_grid = np.asarray(orbital_grid, dtype=int)
    if orbital_grid.shape != (Nx, Ny):
        raise ValueError("orbital_grid shape must match grid_shape")

    def solve(src):
        # Reshape flat orbital source to (Nx, Ny).
        src_2d = np.zeros((Nx, Ny), dtype=float)
        src_2d[:] = src[orbital_grid]
        V_2d = poisson.solve_2d_natural(
            src_2d, dx=dx, dy=dy, eps_r=eps_r, coupling=coupling,
            bc_x_left=bc_x_left, bc_x_right=bc_x_right,
            bc_y_bot=bc_y_bot, bc_y_top=bc_y_top,
        )
        # Return flat V indexed by orbital.
        V_flat = np.zeros(src.size, dtype=float)
        V_flat[orbital_grid] = V_2d
        return V_flat
    return solve


# --------------------------------------------------------------------- #
# The SC loop                                                            #
# --------------------------------------------------------------------- #

def run_sc_loop(
    *,
    make_system,
    n_orbitals,
    poisson_solve,
    mu_per_lead,
    kT=0.0,
    n_background=None,
    mixer: Optional[object] = None,
    tol=1e-5,
    max_iter=80,
    V_init=None,
    density_kw: Optional[dict] = None,
    spin_factor=2,
    verbose=True,
):
    """Run the SC loop until ||ΔV||_∞ < tol.

    Parameters
    ----------
    make_system : callable
        ``make_system(V_orb_flat) -> finalized kwant FiniteSystem``.
        V_orb_flat is a length-``n_orbitals`` array of on-site shifts.
    n_orbitals : int
        Total number of orbitals in the scattering region.
    poisson_solve : callable
        ``poisson_solve(src_flat) -> V_flat`` where ``src_flat`` is
        ``(n_e - n_background)`` reshaped to (n_orbitals,). The callable
        owns the Poisson dimensionality, BCs, and coupling.
    mu_per_lead : sequence of float
        Chemical potential of each lead.
    kT : float
        Electron temperature (same units as energy).
    n_background : (n_orbitals,) array
        Ionic / background electron count per orbital.
    mixer : object with .step(V_in, V_out) method
        Default LinearMixer(0.3).
    tol : float
        Convergence tolerance on max|ΔV| between iterations.
    max_iter : int
    V_init : (n_orbitals,) array, optional
        Starting V (default zero).
    density_kw : dict, optional
        Extra keyword arguments forwarded to the density routine
        (``equilibrium_density_contour`` for the eq case,
        ``nonequilibrium_density_split`` otherwise).
    spin_factor : int
    verbose : bool
    """
    if density_kw is None:
        density_kw = {}
    if mixer is None:
        mixer = mixing.LinearMixer(alpha=0.3)
    if V_init is None:
        V = np.zeros(n_orbitals)
    else:
        V = np.array(V_init, dtype=float).copy()
    if n_background is None:
        n_background = np.zeros(n_orbitals)
    n_background = np.asarray(n_background, dtype=float)
    if n_background.size != n_orbitals:
        raise ValueError("n_background size mismatch")

    residual_history = []
    converged = False
    for it in range(1, max_iter + 1):
        syst = make_system(V)

        if all(np.isclose(m, mu_per_lead[0]) for m in mu_per_lead):
            n_e = density.equilibrium_density_contour(
                syst, mu=mu_per_lead[0], kT=kT, spin_factor=spin_factor,
                **density_kw,
            )
        else:
            n_e = density.nonequilibrium_density_split(
                syst, mu_per_lead=mu_per_lead, kT=kT,
                spin_factor=spin_factor, **density_kw,
            )

        # Poisson source: n_e − n_background (electrons minus ions, in
        # electron-number units). With V_onsite = -eφ this gives the
        # correct screening sign.
        src = n_e - n_background
        V_out = poisson_solve(src)
        if V_out.shape != V.shape:
            raise RuntimeError(
                f"poisson_solve returned shape {V_out.shape}, expected {V.shape}"
            )

        V_new = mixer.step(V, V_out)
        res = float(np.max(np.abs(V_new - V)))
        residual_history.append(res)
        if verbose:
            n_min, n_max = n_e.min(), n_e.max()
            print(f"  iter {it:3d}: max|ΔV| = {res:.3e}   "
                  f"V in [{V.min():+.3f},{V.max():+.3f}]   "
                  f"n in [{n_min:.4f},{n_max:.4f}]")
        V = V_new
        if res < tol:
            converged = True
            break

    return SCResult(V=V, n=n_e, iterations=it, converged=converged,
                    residual_history=residual_history)
