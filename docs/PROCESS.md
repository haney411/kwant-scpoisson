# How this project was built

Chronological account of the work done to turn the empty `~/wrk/kwant`
directory into a working self-consistent Schrödinger-Poisson code on top
of Kwant. Includes the bumps we hit; reading this should make it clear
why the code is shaped the way it is.

## 0. Setup

The starting state was an empty `~/wrk/kwant/` on a dedicated scientific
workstation (system Python 3.12.3, no conda, no venv tooling). User
preferred to install packages straight into the real environment, so we
**configured pip globally** to use `--user --break-system-packages` (via
`~/.config/pip/pip.conf`) — this is the standard escape hatch from the
Debian PEP-668 lockout when you actually do own your Python install.

## 1. Kwant install + smoke test

`pip install kwant` pulled **kwant 1.5.0** plus `tinyarray`. `scipy` was
upgraded to 1.17.1. `matplotlib`, `pytest` installed alongside.

First sanity check: a 1D tight-binding chain with two semi-infinite leads,
asking for the transmission. **This failed**:

```
ValueError: Expected square matrix, got a1.shape=(2, 1)
  at kwant/physics/leads.py:687
```

Root cause: kwant 1.5.0's mode finder calls `scipy.linalg.solve` on a
non-square matrix when all transfer-matrix eigenvalues land in a single
"degenerate cluster" — which happens for a single-orbital lead at
out-of-band energies (all evanescent λ have the same phase, 0 or π).
Newer scipy versions stricten `la.solve` and reject this. Older scipy
silently fell back to lstsq.

We didn't try to patch upstream kwant. Instead we **sidestepped the bug
by design**:

1. The smoke test was restricted to in-band energies — confirmed
   transmission = 1 across the band, validating the install.
2. For SC-Poisson, the equilibrium density is integrated on a **complex
   contour in the upper half plane** — there the integrand is analytic and
   never touches the real-axis band-edge code path. Crucially,
   `lead.selfenergy(z)` *does* work at complex z and at out-of-band real
   z, even though `kwant.smatrix` blows up. So the contour approach
   avoids the bug entirely.

This bug is documented in `notes/kwant_lead_bug.md`.

## 2. NEGF density formulation

Before writing any density code, we wrote `notes/negf_density_formulation.md`
laying out:

- The standard NEGF formula
  ```
  n_i = Σ_α ∫ dE  f_α(E)  ρ_α(i, E),
  ρ_α(i,E) = (1/2π) Σ_n |ψ_n^α(i;E)|²,
  ρ_tot = Σ_α ρ_α   (and LDOS = ρ_tot).
  ```
- A table mapping each quantity to a kwant API call: `kwant.wave_function`
  for per-lead scattering states, `kwant.ldos` for the total LDOS,
  `lead.selfenergy(z)` for the lead self-energy at complex z.
- The decision to use a **complex-contour integration** for the
  equilibrium part of the density and a **real-axis bias-window** for the
  non-equilibrium correction (Brandbyge et al. 2002 split).
- Sign and spin conventions.

The reasoning here mattered because we then knew which kwant primitives
we needed — `kwant.greens_function` was a red herring; what we actually
wanted was `lead.selfenergy(z)` plus our own sparse `(zI − H − Σ)^{-1}`.

## 3. Density module (`scpoisson/density.py`)

Built in roughly this order:

1. `_hamiltonian_sparse`, `_site_orbital_range`, `_interface_orbitals`,
   `_embed_lead_selfenergies` — the plumbing that takes a finalized
   kwant system and gives us `H` and `Σ_total(E)` as sparse matrices over
   the full set of scattering-region orbitals. **One non-trivial point:
   `syst.site_ranges` stores tuples `(first_site_in_block, norbs_per_site,
   first_orbital_in_block)`** — not the obvious "per-site list of orbital
   start indices". The first try got this wrong and the lead self-energy
   was embedded at incorrect orbital rows.
2. `retarded_gf_diagonal(syst, z)` — builds `A = zI − H − Σ_total(z)` and
   solves `A x = e_i` column by column with `splu`, taking only the
   diagonal entries. O(N²) in solves per energy point — fine for the
   small systems of the MVP, expensive at scale. Selected inversion is
   the natural future swap.
3. `ldos_at_energy(syst, E)` — wraps the above as
   `−(1/π) Im diag(G^R)`. Verified to match `kwant.ldos` to machine
   precision (6.75e-16) on a 12-site chain.
4. `equilibrium_density_real_axis` — naive trapezoid LDOS integration.
   Kept around for reference, but the band-edge singularity in the 1D
   chain DOS biases this by ~1% in our tests.
5. `equilibrium_density_contour` — Gauss-Legendre quadrature on a
   semicircular arc in the upper half plane from `E_min` (below the band)
   up to µ. **This is the workhorse.** Validated against the analytic
   1D-chain filling formula `n = 1/2 + arcsin(µ/2t)/π` to machine
   precision across the band.
6. `partial_spectral_function(syst, E, lead)` — per-lead spectral function
   via `kwant.wave_function`. Used in the non-eq correction.
7. `nonequilibrium_density_split` — the Brandbyge-style split:
   equilibrium contour at a reference µ + a small real-axis bias-window
   correction. Stays inside the band of each lead, so the kwant 1.5.0
   bug doesn't bite.

## 4. 1D Poisson + first SC loop

The 1D Poisson solver (`solve_1d_natural`) is a stock flux-conservative
finite-difference Laplacian with Dirichlet or Neumann BCs and
position-dependent ε. Verified against three analytic problems
(constant ρ with Dirichlet, mixed Neumann/Dirichlet, ε jump).

First end-to-end SC loop attempt **failed in three ways** that taught us
the design rules now baked in:

**Failure 1 — wrong sign convention.** We initially set
`rho_e = n_background - n_e` and fed it to `-V'' = coupling·rho_e`. But
the on-site potential in the Hamiltonian is the *electron* potential
energy `V_onsite = −eφ`. The correct source is `n_e − n_background`,
which flips the sign. Symptom: positive extra background charge produced
*positive* V (which would repel electrons rather than attract them) and
electron density went the wrong way.

**Failure 2 — coupling far too strong.** The first run had `coupling=0.3`,
which on a 40-site chain pushes V up to ~30 — completely outside the band
(t=1). Electrons were expelled from the chain. Lesson: for a chain of
length `L = N·dx` and ρ of unit amplitude, the parabolic Poisson solution
scales as `coupling·L²/8`, and we want V to stay O(t) for screening to
make sense — so `coupling ~ 8/N²` is the right scale. This is now
exported as `natural_coupling(N)` in the example.

**Failure 3 — meaningless residual.** Initial residual was
`||ΔV|| / max(||V||, 1e-30)`. When V starts at zero, this divides by
`1e-30` and gives nonsense. Switched to `max|ΔV|` (max-norm absolute),
which has the right semantics: "no site's potential changes by more than
tol between iterations".

After those three fixes, the SC loop converged on all three scenarios
(neutral, small perturbation, doping step) with the right physics.
Anderson mixing was added next; it converged in ~half the iterations
of linear mixing.

## 5. Quasi-2D ribbon — transverse-averaged version first

The first ribbon SC ran with a **transverse-averaged 1D Poisson**: take
the per-column charge `Σ_y (n_e − n_bg)`, solve a 1D Poisson on that, and
broadcast V back uniformly across the transverse direction. This was the
"MVP" version: cheap, easy, and worked.

But it's not enough for the user's actual physics — see next phase.

## 6. Real 2D Poisson and the SC loop refactor

User flagged: "I'm interested in edge effects, electrostatics of the
edges". The transverse-averaged Poisson by construction gives a y-uniform
potential and cannot capture edge band bending. So:

1. **`solve_2d_natural`**: 5-point flux-conservative FD on `(Nx, Ny)` with
   per-edge Dirichlet/Neumann BCs. Includes a gauge pin for the
   all-Neumann singular case. Tested against:
   - 2D Fourier mode (analytic V) → 2e-5 error on 51×41 grid (proper
     O(h²) FD scaling)
   - All-Neumann zero source → V=0 to machine precision
   - Reduction to 1D when source is y-uniform → machine precision match
   - Box Dirichlet with constant ρ vs analytic Fourier series at center
     → 4e-5
2. **Refactored `sc.run_sc_loop`** to take a user-supplied
   `poisson_solve(src_flat) -> V_flat` callable. The BC and coupling
   choices belong to the Poisson backend, not the SC loop. Two factory
   helpers `sc.poisson_solve_1d` and `sc.poisson_solve_2d` cover the
   common cases; users can write their own callable for custom
   geometries.

3. **The 2D ribbon example** then demonstrated the payoff. On a 14×11
   ribbon with uniform jellium background at the bulk-mean density:
   - Reference density has hard-wall edge dip (`n[edge]≈0.137` vs
     `n[bulk]≈0.188`)
   - Full 2D Poisson resolves a y-dependent V profile,
     `|V[edge] − V[center]| ≈ 4e-3` (~40% of mean V)
   - Transverse-averaged 1D gives a y-uniform V → misses the edge effect
     entirely.

A regression test locks this in (`tests/test_ribbon_2d.py`).

## 7. Tests + documentation

21 tests in `tests/`, runnable in one shot via
`python3 tests/run_all.py`. Coverage spans:

- Density: LDOS vs `kwant.ldos`; contour density vs analytic
  1D-chain filling at five values of µ; real-axis with loose tolerance.
- Poisson: 1D Dirichlet/Neumann/ε-jump; 2D Fourier mode; 2D all-Neumann;
  2D-reduces-to-1D; 2D constant ρ vs analytic Fourier series.
- SC loop: neutral fixed point V=0; correct screening sign;
  Anderson convergence.
- Non-equilibrium: zero-bias = equilibrium; clean-chain symmetry under
  bias; asymmetric system shows bias-induced Δn.
- Ribbon (transverse-avg): neutral fixed point; screening sign.
- Ribbon (full 2D): edge band-bending vs y-uniform 1D-avg; neutral fixed
  point.

All passing as of the last full-suite run (21/21).

## Decisions worth flagging

- We did not patch upstream kwant. The mode-finder bug is sidestepped at
  the algorithmic level (contour integration).
- The SC loop is Poisson-backend-agnostic by design (callable interface),
  which makes adding 2D, 3D, or custom solvers a one-function task.
- Performance optimization (parametric Hamiltonians, selected inversion,
  MUMPS) was explicitly deferred — the MVP runs everything in a few
  seconds per SC step, fine for the system sizes we've targeted so far.
- We used the system Python and `pip --user`, no venv (per user
  preference for this dedicated workstation).
