"""End-to-end SC-Poisson on a 1D tight-binding chain.

Three demo scenarios building up complexity:

A) Self-consistent vacuum: n_background = computed n_e(V=0).
   By construction the SC potential is V=0; this validates the loop
   reaches the trivial fixed point.

B) Small charge perturbation: n_background = n_ref + small δ.
   A weak screening potential develops in linear response.

C) Doping step (visible band bending): δ is a step function of moderate
   amplitude across the chain. The screening potential is sigmoid-like.
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


# Natural coupling scale: for a chain of length L=N*dx and ρ of unit
# amplitude, the Poisson solution scales as coupling * L² / 8. To keep
# V ~ O(t)=1 we need coupling ~ 8 / N². For N=30 that's ~0.01.
def natural_coupling(N, target_V_per_unit_rho=1.0):
    return 8.0 * target_V_per_unit_rho / (N * N)


def reference_density(factory, N, mu=0.0):
    """Compute n_e on the V=0 system — our 'neutral' background."""
    V_zero = np.zeros(N)
    syst = factory(V_zero)
    n = density.equilibrium_density_contour(
        syst, mu=mu, kT=0.0, e_min=-3.0, n_arc=60, spin_factor=1,
    )
    return n


def scenario_A_trivial_neutral():
    N = 30
    factory = make_chain_factory(N=N, t=1.0)
    n_ref = reference_density(factory, N, mu=0.0)
    print(f"reference density: bulk={n_ref[5:-5].mean():.6f}  edges={n_ref[0]:.6f},{n_ref[-1]:.6f}")
    coup = natural_coupling(N)
    poisson_solve = sc.poisson_solve_1d(
        dx=1.0, coupling=coup,
        bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
    )
    res = sc.run_sc_loop(
        make_system=factory,
        n_orbitals=N,
        poisson_solve=poisson_solve,
        mu_per_lead=[0.0, 0.0],
        kT=0.0,
        n_background=n_ref,
        mixer=mixing.LinearMixer(alpha=0.5),
        tol=1e-7,
        max_iter=30,
        density_kw=dict(e_min=-3.0, n_arc=50),
        spin_factor=1,
        verbose=True,
    )
    print(f"\nA converged: {res.converged} in {res.iterations} iters")
    print(f"  max|V| = {np.max(np.abs(res.V)):.3e}  (should be ≈0)")
    return res


def scenario_B_small_perturbation():
    N = 30
    factory = make_chain_factory(N=N, t=1.0)
    n_ref = reference_density(factory, N, mu=0.0)
    delta = np.zeros(N)
    delta[N // 2 - 2 : N // 2 + 2] = 0.01
    n_bg = n_ref + delta
    coup = natural_coupling(N)
    print(f"perturbation amplitude: {delta.max():.3f}")
    poisson_solve = sc.poisson_solve_1d(
        dx=1.0, coupling=coup,
        bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
    )
    res = sc.run_sc_loop(
        make_system=factory,
        n_orbitals=N,
        poisson_solve=poisson_solve,
        mu_per_lead=[0.0, 0.0],
        kT=0.0,
        n_background=n_bg,
        mixer=mixing.LinearMixer(alpha=0.5),
        tol=1e-6,
        max_iter=60,
        density_kw=dict(e_min=-3.0, n_arc=50),
        spin_factor=1,
        verbose=True,
    )
    print(f"\nB converged: {res.converged} in {res.iterations} iters")
    print(f"  V profile: {res.V}")
    print(f"  Δn := n_e - n_ref :  {res.n - n_ref}")
    return res


def scenario_C_doping_step():
    N = 40
    factory = make_chain_factory(N=N, t=1.0)
    n_ref = reference_density(factory, N, mu=0.0)
    delta = np.zeros(N)
    delta[N // 2:] = 0.05
    n_bg = n_ref + delta
    coup = natural_coupling(N)
    poisson_solve = sc.poisson_solve_1d(
        dx=1.0, coupling=coup,
        bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
    )
    res = sc.run_sc_loop(
        make_system=factory,
        n_orbitals=N,
        poisson_solve=poisson_solve,
        mu_per_lead=[0.0, 0.0],
        kT=0.0,
        n_background=n_bg,
        mixer=mixing.LinearMixer(alpha=0.3),
        tol=1e-5,
        max_iter=100,
        density_kw=dict(e_min=-3.0, n_arc=50),
        spin_factor=1,
        verbose=True,
    )
    print(f"\nC converged: {res.converged} in {res.iterations} iters")
    print(f"  V min/max: {res.V.min():+.4f} {res.V.max():+.4f}")
    print(f"  Δn := n_e - n_ref :  {res.n - n_ref}")
    return res


if __name__ == "__main__":
    print("=== Scenario A: trivial neutral ===")
    rA = scenario_A_trivial_neutral()

    print("\n=== Scenario B: small perturbation ===")
    rB = scenario_B_small_perturbation()

    print("\n=== Scenario C: doping step ===")
    rC = scenario_C_doping_step()
