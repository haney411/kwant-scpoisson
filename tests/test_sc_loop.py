"""Integration tests for the SC-Poisson loop.

These exercise the full equilibrium loop on a 1D chain and verify a few
qualitative + quantitative properties.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kwant
from scpoisson import sc, density, mixing


def make_chain_factory(N, t=1.0):
    lat = kwant.lattice.chain(norbs=1)
    sym = kwant.TranslationalSymmetry((-1,))

    def factory(V_onsite):
        syst = kwant.Builder()
        for i in range(N):
            syst[lat(i)] = float(V_onsite[i])
        for i in range(N - 1):
            syst[lat(i), lat(i + 1)] = -t
        lead = kwant.Builder(sym)
        lead[lat(0)] = 0.0
        lead[lat(0), lat(1)] = -t
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
        return syst.finalized()

    return factory


def reference_density(factory, N, mu=0.0):
    syst = factory(np.zeros(N))
    return density.equilibrium_density_contour(
        syst, mu=mu, kT=0.0, e_min=-3.0, n_arc=60, spin_factor=1,
    )


def _poisson_1d(N, coupling):
    return sc.poisson_solve_1d(
        dx=1.0, coupling=coupling,
        bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
    )


def test_neutral_fixed_point_is_zero():
    """When n_background == n_e(V=0), the SC fixed point is V=0."""
    N = 20
    factory = make_chain_factory(N)
    n_ref = reference_density(factory, N)
    coup = 8.0 / N**2
    res = sc.run_sc_loop(
        make_system=factory, n_orbitals=N,
        poisson_solve=_poisson_1d(N, coup),
        mu_per_lead=[0.0, 0.0], kT=0.0,
        n_background=n_ref,
        mixer=mixing.LinearMixer(0.5), tol=1e-8, max_iter=20,
        density_kw=dict(e_min=-3.0, n_arc=40),
        spin_factor=1, verbose=False,
    )
    assert res.converged
    assert np.max(np.abs(res.V)) < 1e-10


def test_screening_response_sign():
    """Extra positive bg → negative V → extra electrons. Verify sign."""
    N = 30
    factory = make_chain_factory(N)
    n_ref = reference_density(factory, N)
    delta = np.zeros(N)
    delta[N // 2 - 2 : N // 2 + 2] = 0.01
    coup = 8.0 / N**2
    res = sc.run_sc_loop(
        make_system=factory, n_orbitals=N,
        poisson_solve=_poisson_1d(N, coup),
        mu_per_lead=[0.0, 0.0], kT=0.0,
        n_background=n_ref + delta,
        mixer=mixing.LinearMixer(0.5), tol=1e-6, max_iter=40,
        density_kw=dict(e_min=-3.0, n_arc=40),
        spin_factor=1, verbose=False,
    )
    assert res.converged
    middle = res.V[N // 2 - 2 : N // 2 + 2]
    assert np.all(middle < 0)
    dn = res.n - n_ref
    middle_dn = dn[N // 2 - 2 : N // 2 + 2]
    assert np.all(middle_dn > 0)


def test_anderson_mixer_converges():
    """Anderson mixer should also converge on the same problem."""
    N = 30
    factory = make_chain_factory(N)
    n_ref = reference_density(factory, N)
    delta = np.zeros(N)
    delta[N // 2 - 2 : N // 2 + 2] = 0.01
    coup = 8.0 / N**2
    res = sc.run_sc_loop(
        make_system=factory, n_orbitals=N,
        poisson_solve=_poisson_1d(N, coup),
        mu_per_lead=[0.0, 0.0], kT=0.0,
        n_background=n_ref + delta,
        mixer=mixing.AndersonMixer(alpha=0.5, history=5),
        tol=1e-6, max_iter=40,
        density_kw=dict(e_min=-3.0, n_arc=40),
        spin_factor=1, verbose=False,
    )
    assert res.converged, f"Anderson did not converge in {res.iterations} iters"
    print(f"  Anderson converged in {res.iterations} iters")


if __name__ == "__main__":
    print("test_neutral_fixed_point_is_zero ...")
    test_neutral_fixed_point_is_zero()
    print("PASS")
    print("test_screening_response_sign ...")
    test_screening_response_sign()
    print("PASS")
    print("test_anderson_mixer_converges ...")
    test_anderson_mixer_converges()
    print("PASS")
    print("\nAll SC-loop tests passed.")
