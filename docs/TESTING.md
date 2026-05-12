# Testing and validation

Strategy: every piece of physics in the codebase has at least one test
that compares against either an analytic reference, a kwant primitive, or
an independent path through our own code. We avoid "the code passes if
nothing changes" snapshot tests; every assertion is a substantive
correctness check.

Run everything: `python3 tests/run_all.py`

Current state: **21 tests, 0 failures.**

## Test files

```
tests/
├── run_all.py           # imports all test modules, runs every test_*()
├── test_density.py      # 4 tests on the NEGF density layer
├── test_poisson.py      # 7 tests on the Poisson solvers
├── test_sc_loop.py      # 3 tests on the SC iteration
├── test_nonequilibrium.py  # 3 tests on the non-eq density path
├── test_ribbon.py       # 2 tests on transverse-averaged ribbon SC
└── test_ribbon_2d.py    # 2 tests on the full-2D-Poisson ribbon
```

## Density (`test_density.py`)

**`test_ldos_matches_kwant`** — compute LDOS at E=0.5 on a 12-site
chain two ways: our `density.ldos_at_energy` (via the
`(zI−H−Σ)^{-1}` diagonal) vs `kwant.ldos`. Agreement to **6.75e-16
relative**.

**`test_uniform_chain_density_half_filling_real_axis`** — real-axis
trapezoid integration of LDOS up to µ=0 on a uniform chain. Bulk density
should approach 0.5 (spinless half-filling). Actual: 0.493, off by ~1.5%
due to the band-edge 1/√(4t²−E²) singularity that trapezoid mishandles.
The test asserts < 2% bulk error.

**`test_contour_density_matches_analytic`** — for the same chain,
`equilibrium_density_contour` should match the analytic filling
`n = 1/2 + arcsin(µ/2t)/π` exactly. Five values of µ ∈ {−1, −0.5, 0,
+0.5, +1}. Result: **agreement to machine precision (errors ~5e-16)**
across the board. This is the most important density test — it locks in
that the entire contour pipeline (lead self-energy at complex z, sparse
G^R diagonal, Gauss-Legendre integration) is correct.

**`test_real_axis_density_matches_analytic_loosely`** — sanity for the
real-axis path: same problem, < 2% error. Currently 0.88% on a 3000-point
mesh; the limit is the band-edge singularity, not the mesh.

## Poisson (`test_poisson.py`)

**1D solver — analytic comparisons:**

- `test_constant_rho_dirichlet`: `−V''=1`, V(0)=V(L)=0 → `V=x(L−x)/2`.
  Match to **1.25e-15** (machine precision; FD is exact for quadratics).
- `test_neumann_dirichlet_mix`: `−V''=1`, `V'(0)=0`, `V(L)=V_R` →
  `V=V_R + (L²−x²)/2`. Match to 2.5e-3 — first-order one-sided Neumann
  costs O(dx) accuracy near the boundary.
- `test_piecewise_eps_jump`: ε=1 in left half, ε=2 in right half, ρ=1,
  Dirichlet zero. Sanity: V>0 in the interior, finite, max in expected
  region.

**2D solver — analytic comparisons:**

- `test_2d_fourier_mode_dirichlet`: `ρ = sin(πx/Lx)·sin(πy/Ly)` with all
  zero-Dirichlet boundaries → `V = ρ / (π²/Lx² + π²/Ly²)`. On a 51×41
  grid: **2.1e-5 max error**, fully consistent with O(h²) FD.
- `test_2d_neumann_zero_source`: zero source, all-Neumann, gauge pin
  → V≡0 exactly.
- `test_2d_reduces_to_1d`: y-uniform ρ + Neumann y-edges → V should be
  y-uniform and match the 1D solver column-by-column. Match: **6e-16
  relative** (machine precision); y-variation of the 2D answer is 3.6e-17
  (i.e., zero up to round-off).
- `test_2d_dirichlet_uniform_eps_const_rho`: constant ρ in the unit
  square with all-zero Dirichlet. Compare V at center to the series
  solution `(16/π⁴) Σ_{m,n odd} 1/(mn(m²+n²))`. Match: 4e-5 (numerical
  ≈ 0.07364, analytic ≈ 0.07368).

These together validate the linear algebra, the BC handling for all
combinations, and the reduction to known limits.

## SC loop (`test_sc_loop.py`)

**`test_neutral_fixed_point_is_zero`** — when `n_background` is set to
`n_e(V=0)`, the SC iteration's fixed point should be V=0 exactly (no net
charge anywhere). Convergence in 1 iteration with linear mixing,
max|V| < 1e-10.

**`test_screening_response_sign`** — extra positive bg in the middle of
a 30-site chain should produce V<0 (electron potential energy is
*lowered* where positive charge sits) and Δn>0 (electrons accumulate to
screen). Both signs asserted.

**`test_anderson_mixer_converges`** — same problem as above, but with
`AndersonMixer`. Should converge (and faster — typically 4 iters vs 10
for linear).

## Non-equilibrium (`test_nonequilibrium.py`)

**`test_zero_bias_matches_equilibrium`** — for `µ_L = µ_R = 0`, the
non-equilibrium-split density should match the equilibrium contour
density bitwise. Result: **0 difference**.

**`test_clean_chain_symmetric_bias_invariant`** — symmetric clean 1D
chain at bias V=0.3 (`µ_L=+0.15`, `µ_R=−0.15`). By reflection symmetry,
each lead's spectral function is half the LDOS, so the bias just
reshuffles flow without changing density at µ_ref=0. Result: **diff = 0**.

**`test_asymmetric_chain_bias_changes_density`** — chain with an
attractive on-site well in the middle, then bias µ_L=+0.4, µ_R=−0.4.
Asymmetric scatterer → bias-induced density change. Verified that
running the non-eq density routine doesn't crash and produces a
non-trivial answer (1.6e-2 max diff from eq).

## Ribbon, transverse-averaged Poisson (`test_ribbon.py`)

**`test_ribbon_neutral_zero_perturbation`** — zero δ-doping on a 10×3
ribbon → V=0 exactly to within 1e-8.

**`test_ribbon_screening_response_sign`** — step doping on the right
half of a 12×3 ribbon. V more negative on the right, Δn more positive on
the right (correct screening signs).

## Ribbon, full 2D Poisson (`test_ribbon_2d.py`)

**`test_2d_poisson_resolves_edge_bending`** — the **headline test of the
2D path.** Wide ribbon (10×11), uniform jellium at bulk-mean. The
hard-wall confinement gives the reference density a deficit at the
transverse edges (`n[edge] < n[bulk]`). Full 2D Poisson should develop
a y-dependent V profile to screen this:

- Assertion 1: `|V[mid, edge] − V[mid, center]| > 1e−4` (the y-structure
  is real, not numerical noise). Measured: **5e-3**.
- Assertion 2: same problem with the transverse-averaged 1D Poisson
  should give a y-uniform V (broadcast). Measured y-std: **0** exactly.

This is the test that proves the 2D path captures edge electrostatics
that the 1D path cannot.

**`test_2d_poisson_neutral_with_self_reference`** — analogue of the
1D neutral fixed-point test on the ribbon: setting `n_background =
n_e(V=0)` with matching `density_kw` between reference and SC run
should give V→0. Max|V| after 10 iters < 1e-9.

## Tests we did *not* write (and why)

- **Snapshot tests of full V/n arrays.** These tend to brittle-break on
  innocuous refactors and don't add correctness coverage when an analytic
  reference exists.
- **Performance benchmarks.** Out of scope for the MVP. We measured
  speed informally (Anderson converged in ~half the iters of linear) but
  didn't make it a regression target.
- **Tests of the kwant install itself.** The smoke test in
  `scratch/smoke_test.py` lives outside `tests/`; we treat the kwant
  install as a precondition.

## Reproducibility

Every test is deterministic (no `np.random`). The expected pass rate is
21/21. Any deviation means something changed substantively — the failing
test will tell you whether it's a physics regression (sign, scaling,
band-edge handling) or a numerical-tolerance regression (different
scipy/kwant versions, etc.).

Run on dependency upgrades. Re-run after any modification to
`density.py`, `poisson.py`, or `sc.py`.

## What still wants more coverage

- We do not yet have a test that **integrates the Poisson source globally
  and checks charge balance** in the all-Neumann case. (For an isolated
  system with no Dirichlet anywhere, the total source must integrate to
  the boundary flux for a solution to exist; we should assert that or at
  least warn.)
- We have no test of **finite-T** density at temperatures where the
  Fermi-window correction is non-trivial. The finite-T contour treatment
  is approximate; a temperature scan against the T=0 limit would be a
  useful regression.
- We have no test of a **multi-orbital site** (e.g., spinful
  Hamiltonian with norbs=2 in kwant). The orbital-range helpers handle
  this case in principle but it is exercised only by the single-orbital
  code paths in the current test suite.
