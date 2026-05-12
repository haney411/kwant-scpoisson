"""Tests for scpoisson.density.

Sanity checks against analytic / kwant cross-references:

1. `ldos_at_energy` agrees with `kwant.ldos` at in-band energies.
2. Equilibrium density of a uniform 1D chain at µ=0, T=0 equals 0.5 per
   site (spin_factor=1) in the bulk.
3. Real-axis and contour density methods agree on a uniform chain.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kwant
from scpoisson import density


def make_chain(N=20, t=1.0, V_onsite=None):
    """Uniform 1D chain of length N with two semi-infinite leads.

    Optional ``V_onsite`` array (length N) is added to the on-site energy
    of each scattering-region site.
    """
    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    if V_onsite is None:
        V_onsite = np.zeros(N)
    for i in range(N):
        syst[lat(i)] = float(V_onsite[i])
    for i in range(N - 1):
        syst[lat(i), lat(i + 1)] = -t
    sym = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym)
    lead[lat(0)] = 0.0
    lead[lat(0), lat(1)] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()


def test_ldos_matches_kwant():
    syst = make_chain(N=12, t=1.0)
    E = 0.5
    my_ldos = density.ldos_at_energy(syst, E)
    kw_ldos = kwant.ldos(syst, E)
    assert my_ldos.shape == kw_ldos.shape
    print(f"  my LDOS at E={E}: {my_ldos}")
    print(f"  kw LDOS at E={E}: {kw_ldos}")
    rel = np.max(np.abs(my_ldos - kw_ldos)) / np.max(np.abs(kw_ldos))
    print(f"  max rel diff: {rel:.3e}")
    assert np.allclose(my_ldos, kw_ldos, rtol=1e-8, atol=1e-12), \
        f"LDOS mismatch (max abs diff = {np.max(np.abs(my_ldos-kw_ldos))})"


def test_uniform_chain_density_half_filling_real_axis():
    """Bulk density of uniform 1D chain at µ=0 should be ~0.5 per site
    (no spin, single-orbital). Edges have lead-induced deviations."""
    N = 40
    syst = make_chain(N=N, t=1.0)
    n = density.equilibrium_density_real_axis(
        syst, mu=0.0, kT=0.0, n_energy=2000,
        e_min=-2.0 + 1e-3, e_max=0.0,
        spin_factor=1,
    )
    print(f"  density profile: {n}")
    # Look at the bulk (middle 20 sites)
    bulk = n[10:-10]
    print(f"  bulk mean: {bulk.mean():.6f} (expect 0.5)")
    print(f"  bulk std:  {bulk.std():.6f}")
    assert abs(bulk.mean() - 0.5) < 0.02, \
        f"bulk density {bulk.mean()} far from 0.5"


def test_contour_density_matches_analytic():
    """Contour integration matches the analytic 1D chain filling.

    For H = -t Σ (c†_i c_{i+1} + h.c.) with t=1 and µ in (-2t, 2t):
      n_bulk(µ) = 1/2 + (1/π) arcsin(µ/2t)         (spinless, single-orbital)
    """
    N = 30
    syst = make_chain(N=N, t=1.0)
    for mu in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        n_co = density.equilibrium_density_contour(
            syst, mu=mu, kT=0.0, e_min=-3.0, n_arc=60, spin_factor=1,
        )
        analytic = 0.5 + np.arcsin(mu / 2.0) / np.pi
        bulk = n_co[5:-5].mean()
        err = abs(bulk - analytic)
        print(f"  µ={mu:+.2f}:  analytic={analytic:.6f}  contour bulk={bulk:.6f}  err={err:.2e}")
        assert err < 5e-4, f"contour bulk {bulk} vs analytic {analytic}, err {err}"


def test_real_axis_density_matches_analytic_loosely():
    """Real-axis trapezoid is biased near band edges; allow ~1% error."""
    N = 30
    syst = make_chain(N=N, t=1.0)
    mu = -0.5
    n_ra = density.equilibrium_density_real_axis(
        syst, mu=mu, kT=0.0, n_energy=3000,
        e_min=-2.0 + 1e-3, e_max=mu, spin_factor=1,
    )
    analytic = 0.5 + np.arcsin(mu / 2.0) / np.pi
    bulk = n_ra[5:-5].mean()
    err = abs(bulk - analytic)
    print(f"  real-axis bulk={bulk:.6f} vs analytic={analytic:.6f} err={err:.2e}")
    assert err < 0.02, f"real-axis off by {err}"


if __name__ == "__main__":
    print("test_ldos_matches_kwant ...")
    test_ldos_matches_kwant()
    print("PASS")
    print()
    print("test_uniform_chain_density_half_filling_real_axis ...")
    test_uniform_chain_density_half_filling_real_axis()
    print("PASS")
    print()
    print("test_contour_density_matches_analytic ...")
    test_contour_density_matches_analytic()
    print("PASS")
    print()
    print("test_real_axis_density_matches_analytic_loosely ...")
    test_real_axis_density_matches_analytic_loosely()
    print("PASS")
    print()
    print("All density tests passed.")
