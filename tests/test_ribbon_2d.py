"""Regression test for the 2D-Poisson ribbon example.

Verifies that the full 2D Poisson resolves a y-dependent edge band
bending that the transverse-averaged 1D Poisson misses.
"""

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples"))

import numpy as np
import example_sc_ribbon_2d as ribbon2d


def test_2d_poisson_resolves_edge_bending():
    r = ribbon2d.run_2d_vs_1d(Nx=10, Ny=11, mu=2.0, perturbation=0.0)
    V_2d = r["V_2d"]
    V_1d = r["V_1d"]
    Nx, Ny = V_2d.shape
    # 2D V at x=mid should differ between edge and center
    edge_v = V_2d[Nx // 2, 0]
    cent_v = V_2d[Nx // 2, Ny // 2]
    delta_2d = abs(edge_v - cent_v)
    # 1D-averaged V is broadcast in y so column should be uniform
    delta_1d = float(V_1d[Nx // 2, :].std())
    print(f"  |V[edge]-V[center]| (2D): {delta_2d:.3e}")
    print(f"  V column std (1D-avg):    {delta_1d:.3e}")
    assert delta_2d > 1e-4, "2D Poisson failed to develop y-structure"
    assert delta_1d < 1e-12, "1D-avg Poisson should be y-uniform but isn't"


def test_2d_poisson_neutral_with_self_reference():
    """If n_background = n_e(V=0) (per orbital), V should converge to ~0
    even in the 2D case (this checks the 2D loop self-consistency)."""
    Nx, Ny = 8, 7
    from scpoisson import sc, density, mixing
    factory = ribbon2d.make_ribbon_factory(Nx=Nx, Ny=Ny)
    syst0 = factory(np.zeros(Nx * Ny))
    grid = ribbon2d.orbital_grid(syst0, Nx, Ny)
    density_kw = dict(e_min=-1.0, n_arc=80)
    n_ref = density.equilibrium_density_contour(
        syst0, mu=2.0, kT=0.0, spin_factor=1, **density_kw,
    )
    coup = 8.0 / (Nx * Ny)
    poisson_2d = sc.poisson_solve_2d(
        grid_shape=(Nx, Ny), orbital_grid=grid, dx=1.0, dy=1.0,
        coupling=coup, eps_r=1.0,
        bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
        bc_y_bot=("neumann", 0.0), bc_y_top=("neumann", 0.0),
    )
    res = sc.run_sc_loop(
        make_system=factory, n_orbitals=Nx * Ny,
        poisson_solve=poisson_2d,
        mu_per_lead=[2.0, 2.0], kT=0.0,
        n_background=n_ref,
        mixer=mixing.LinearMixer(0.5),
        tol=1e-7, max_iter=10,
        density_kw=density_kw,
        spin_factor=1, verbose=False,
    )
    assert res.converged
    assert np.max(np.abs(res.V)) < 1e-9, \
        f"V should be ~0 when n_bg matches n_ref; got {np.max(np.abs(res.V)):.3e}"


if __name__ == "__main__":
    print("test_2d_poisson_resolves_edge_bending ...")
    test_2d_poisson_resolves_edge_bending()
    print("PASS\n")
    print("test_2d_poisson_neutral_with_self_reference ...")
    test_2d_poisson_neutral_with_self_reference()
    print("PASS\n")
    print("All 2D-ribbon tests passed.")
