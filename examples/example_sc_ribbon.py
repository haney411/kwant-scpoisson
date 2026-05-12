"""SC-Poisson on a quasi-2D ribbon (finite transverse width).

The system is a strip of W sites in the transverse (y) direction and N
sites longitudinally (x). Hard-wall boundary in y. Leads are
semi-infinite ribbons in ±x.

For Poisson we use a *transverse-averaged* 1D model: the longitudinal
potential V(x) is the same on every transverse site at column x. The
source for V(x) is the column-summed electron density n(x) = Σ_y n(x,y).

(Future: replace with a full 2D Poisson solver. For now this is a clean
MVP that already exposes the multi-mode physics of the ribbon.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kwant
from scpoisson import sc, density, mixing


def make_ribbon_factory(N, W, t=1.0):
    """Return a closure: V_col_array (length N) -> finalized ribbon system.

    All sites in column x get the same on-site shift V_col[x]. This is the
    transverse-average ansatz: the Poisson potential depends only on x.
    """
    lat = kwant.lattice.square(norbs=1)
    sym = kwant.TranslationalSymmetry((-1, 0))

    def factory(V_col):
        syst = kwant.Builder()
        for x in range(N):
            for y in range(W):
                syst[lat(x, y)] = 4.0 * t + float(V_col[x])
                # +4t shifts the on-site so the 2D band runs ~[0, 8t].
                # Actually for a 2D square lattice with t hopping, band is
                # ±4t around the on-site. We keep on-site at +4t so the
                # band runs [0, 8t]; bottom of band at E=0.
        for x in range(N):
            for y in range(W - 1):
                syst[lat(x, y), lat(x, y + 1)] = -t   # transverse
        for x in range(N - 1):
            for y in range(W):
                syst[lat(x, y), lat(x + 1, y)] = -t   # longitudinal
        # Leads: ribbon with the same transverse cross-section.
        lead = kwant.Builder(sym)
        for y in range(W):
            lead[lat(0, y)] = 4.0 * t
        for y in range(W - 1):
            lead[lat(0, y), lat(0, y + 1)] = -t
        for y in range(W):
            lead[lat(0, y), lat(1, y)] = -t
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
        return syst.finalized()

    return factory


def column_map(syst, N, W):
    """Return an (N, W) array of orbital indices: orb = column_map[x, y]."""
    cmap = np.full((N, W), -1, dtype=int)
    for orb_idx, site in enumerate(syst.sites):
        x, y = site.tag
        if 0 <= x < N and 0 <= y < W:
            cmap[x, y] = orb_idx
    if (cmap < 0).any():
        raise RuntimeError("not every (x,y) mapped to an orbital")
    return cmap


def transverse_summed_density(n_orb, cmap):
    """n_col[x] = Σ_y n_orb[ cmap[x,y] ]."""
    return n_orb[cmap].sum(axis=1)


def run_ribbon_sc(N=15, W=4, mu=2.0, coupling_scale=None,
                  delta_bg_strength=0.05, max_iter=60):
    """Run an SC-Poisson on a ribbon with a step doping profile.

    The Fermi level µ is inside the ribbon's lowest few subbands.
    """
    t = 1.0
    factory = make_ribbon_factory(N=N, W=W, t=t)
    syst0 = factory(np.zeros(N))
    cmap = column_map(syst0, N, W)

    # Reference electron density at V=0 (column-summed):
    n_ref_orb = density.equilibrium_density_contour(
        syst0, mu=mu, kT=0.0, e_min=-1.0, n_arc=80, spin_factor=1,
    )
    n_ref_col = transverse_summed_density(n_ref_orb, cmap)
    print(f"reference n_col (per column): {n_ref_col}")
    print(f"  bulk(col) mean = {n_ref_col[N//4:-N//4].mean():.4f}")

    # Doping perturbation in the column-summed (per-column) charge:
    delta_col = np.zeros(N)
    delta_col[N // 2:] = delta_bg_strength * W   # uniform across transverse
    n_bg_col = n_ref_col + delta_col

    # Custom SC loop wrapper that runs Poisson on per-column charge.
    if coupling_scale is None:
        coupling_scale = 8.0 / (N * N) / W   # normalize so V ~ O(t)

    # We piggyback on sc.run_sc_loop by giving it a system with a
    # density-collapse: replace the density function to return per-column n.
    # Cleanest path: implement the loop here directly.
    V_col = np.zeros(N)
    mixer = mixing.AndersonMixer(alpha=0.5, history=5)
    residuals = []
    converged = False
    from scpoisson import poisson
    for it in range(1, max_iter + 1):
        syst = factory(V_col)
        n_orb = density.equilibrium_density_contour(
            syst, mu=mu, kT=0.0, e_min=-1.0, n_arc=80, spin_factor=1,
        )
        n_col = transverse_summed_density(n_orb, cmap)
        src = n_col - n_bg_col
        V_out = poisson.solve_1d_natural(
            src, dx=1.0, eps_r=1.0, coupling=coupling_scale,
            bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
        )
        V_new = mixer.step(V_col, V_out)
        r = float(np.max(np.abs(V_new - V_col)))
        residuals.append(r)
        print(f"  iter {it:3d}: max|ΔV| = {r:.3e}   V in [{V_col.min():+.3f},{V_col.max():+.3f}]   n_col mid = {n_col[N//2]:.4f}")
        V_col = V_new
        if r < 1e-5:
            converged = True
            break
    print(f"\nConverged: {converged} in {it} iters")
    print(f"V_col final: {V_col}")
    print(f"n_col final: {n_col}")
    print(f"Δn_col := n_col - n_ref_col: {n_col - n_ref_col}")
    return V_col, n_col, n_ref_col


if __name__ == "__main__":
    print("=== Ribbon SC (N=15, W=4, µ=2.0) ===")
    V, n_col, n_ref = run_ribbon_sc(N=15, W=4, mu=2.0, delta_bg_strength=0.05)
