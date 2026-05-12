"""SC-Poisson on a quasi-2D ribbon with **full 2D Poisson** (per (x,y) site).

Goal: resolve edge effects. The transverse-averaged 1D Poisson cannot
distinguish bulk from edge in y; the full 2D solver can.

Setup:
- Square-lattice ribbon, Nx longitudinal × Ny transverse sites.
- Hard-wall y boundaries in the tight-binding model.
- 2D Poisson with Dirichlet on x ends (V=0, lead reference) and Neumann
  on y edges (∂V/∂y = 0, free-standing ribbon, no normal flux).
- Background doping: choose `n_bg = bulk average of n_e(V=0)`. This is
  the *uniform-jellium* background: the bulk SC potential is then ≈ 0,
  but at the transverse edges where the equilibrium electron density
  differs from the bulk average, the SC develops a non-trivial V(y).

The example also runs a transverse-averaged ("1D" Poisson) calculation
on the same problem and prints both V profiles for comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kwant
from scpoisson import sc, density, mixing, poisson


# --------------------------------------------------------------------- #
# System factory                                                         #
# --------------------------------------------------------------------- #

def make_ribbon_factory(Nx, Ny, t=1.0, on_site_base=4.0):
    """Closure: V_flat (length Nx*Ny) -> finalized ribbon system.

    on_site_base shifts the band so it sits roughly at [0, 8t] (2D square
    lattice with t hopping has band ±4t around on-site).
    """
    lat = kwant.lattice.square(norbs=1)
    sym = kwant.TranslationalSymmetry((-1, 0))

    def factory(V_flat):
        V_flat = np.asarray(V_flat, dtype=float)
        syst = kwant.Builder()
        for x in range(Nx):
            for y in range(Ny):
                # orbital index = x*Ny + y (matches grid mapping below)
                k = x * Ny + y
                syst[lat(x, y)] = on_site_base * t + float(V_flat[k])
        for x in range(Nx):
            for y in range(Ny - 1):
                syst[lat(x, y), lat(x, y + 1)] = -t
        for x in range(Nx - 1):
            for y in range(Ny):
                syst[lat(x, y), lat(x + 1, y)] = -t
        # Leads: ribbon with same transverse cross-section.
        lead = kwant.Builder(sym)
        for y in range(Ny):
            lead[lat(0, y)] = on_site_base * t
        for y in range(Ny - 1):
            lead[lat(0, y), lat(0, y + 1)] = -t
        for y in range(Ny):
            lead[lat(0, y), lat(1, y)] = -t
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
        return syst.finalized()

    return factory


def orbital_grid(syst, Nx, Ny):
    """Return an (Nx, Ny) int array: grid[x, y] = orbital index of site (x, y)."""
    g = np.full((Nx, Ny), -1, dtype=int)
    for orb_idx, site in enumerate(syst.sites):
        x, y = site.tag
        if 0 <= x < Nx and 0 <= y < Ny:
            g[x, y] = orb_idx
    if (g < 0).any():
        raise RuntimeError("not every (x,y) mapped")
    return g


def flat_to_grid(arr_flat, grid):
    Nx, Ny = grid.shape
    arr_2d = np.empty((Nx, Ny), dtype=arr_flat.dtype)
    arr_2d[:] = arr_flat[grid]
    return arr_2d


# --------------------------------------------------------------------- #
# Run                                                                    #
# --------------------------------------------------------------------- #

def run_2d_vs_1d(Nx=12, Ny=11, mu=2.0, perturbation=0.0):
    """Run SC-Poisson on a ribbon with both 2D and transverse-avg 1D
    Poisson; return both results for comparison.
    """
    t = 1.0
    factory = make_ribbon_factory(Nx=Nx, Ny=Ny, t=t)
    syst0 = factory(np.zeros(Nx * Ny))
    grid = orbital_grid(syst0, Nx, Ny)
    n_orb = Nx * Ny

    # Reference density at V=0 (per orbital):
    n_ref = density.equilibrium_density_contour(
        syst0, mu=mu, kT=0.0, e_min=-1.0, n_arc=80, spin_factor=1,
    )
    n_ref_2d = flat_to_grid(n_ref, grid)
    n_bulk_avg = n_ref_2d[Nx // 4 : -Nx // 4, Ny // 2].mean()
    print(f"reference n bulk-average ≈ {n_bulk_avg:.4f}")
    print("reference n profile across transverse (middle column):")
    print("  ", n_ref_2d[Nx // 2, :])

    # Background: uniform jellium at bulk-mean.
    n_bg = np.full(n_orb, n_bulk_avg)
    # + optional uniform perturbation
    n_bg = n_bg + perturbation

    # ---- Run with full 2D Poisson ----
    print("\n--- 2D Poisson run ---")
    coup_2d = 8.0 / (Nx * Ny)   # natural scale
    poisson_2d = sc.poisson_solve_2d(
        grid_shape=(Nx, Ny), orbital_grid=grid, dx=1.0, dy=1.0,
        coupling=coup_2d, eps_r=1.0,
        bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
        bc_y_bot=("neumann", 0.0), bc_y_top=("neumann", 0.0),
    )
    res2d = sc.run_sc_loop(
        make_system=factory, n_orbitals=n_orb,
        poisson_solve=poisson_2d,
        mu_per_lead=[mu, mu], kT=0.0,
        n_background=n_bg,
        mixer=mixing.AndersonMixer(alpha=0.5, history=5),
        tol=1e-5, max_iter=40,
        density_kw=dict(e_min=-1.0, n_arc=70),
        spin_factor=1, verbose=True,
    )
    V_2d = flat_to_grid(res2d.V, grid)
    n_2d = flat_to_grid(res2d.n, grid)

    # ---- Run with transverse-averaged ("1D") Poisson ----
    print("\n--- Transverse-averaged 1D Poisson run ---")
    def poisson_1d_avg(src_flat):
        src_2d = flat_to_grid(src_flat, grid)
        src_col = src_2d.sum(axis=1)
        # 1D Poisson on Nx grid → V_col (length Nx)
        V_col = poisson.solve_1d_natural(
            src_col, dx=1.0, eps_r=1.0,
            coupling=8.0 / Nx**2 / Ny,
            bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
        )
        # Broadcast back to (Nx, Ny) flat
        V_flat = np.zeros(n_orb)
        for x in range(Nx):
            for y in range(Ny):
                V_flat[grid[x, y]] = V_col[x]
        return V_flat

    res1d = sc.run_sc_loop(
        make_system=factory, n_orbitals=n_orb,
        poisson_solve=poisson_1d_avg,
        mu_per_lead=[mu, mu], kT=0.0,
        n_background=n_bg,
        mixer=mixing.AndersonMixer(alpha=0.5, history=5),
        tol=1e-5, max_iter=40,
        density_kw=dict(e_min=-1.0, n_arc=70),
        spin_factor=1, verbose=True,
    )
    V_1d = flat_to_grid(res1d.V, grid)

    # ---- Comparisons ----
    print("\n=== Edge vs bulk comparison (2D Poisson) ===")
    print(f"V at x=mid, y profile (2D): {V_2d[Nx // 2, :]}")
    print(f"V at x=mid, y profile (1D-avg, broadcast): {V_1d[Nx // 2, :]}")
    print()
    print(f"V[mid, edge]    = {V_2d[Nx // 2, 0]:+.5f}, {V_2d[Nx // 2, -1]:+.5f}")
    print(f"V[mid, center]  = {V_2d[Nx // 2, Ny // 2]:+.5f}")
    print(f"Δ(edge-center) (2D):   {V_2d[Nx // 2, 0] - V_2d[Nx // 2, Ny // 2]:+.5e}")
    print(f"Δ(edge-center) (1D-avg): {V_1d[Nx // 2, 0] - V_1d[Nx // 2, Ny // 2]:+.5e}  (should be 0 by construction)")
    print()
    print("Per-row Δn (2D, x=mid): ", n_2d[Nx // 2, :] - n_ref_2d[Nx // 2, :])

    return {
        "grid": grid, "n_ref_2d": n_ref_2d,
        "V_2d": V_2d, "n_2d": n_2d, "iters_2d": res2d.iterations,
        "V_1d": V_1d, "iters_1d": res1d.iterations,
    }


if __name__ == "__main__":
    result = run_2d_vs_1d(Nx=14, Ny=11, mu=2.0, perturbation=0.0)
