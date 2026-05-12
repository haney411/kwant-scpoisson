# Architecture

The `scpoisson` package, layer by layer.

```
                ┌───────────────────────────────────┐
                │   user code (factory, n_bg, ...)  │
                └────────────────┬──────────────────┘
                                 ▼
                ┌───────────────────────────────────┐
                │  sc.run_sc_loop                   │
                │   ┌──────────────────────────┐    │
                │   │ density.equilibrium_     │    │
                │   │ density_contour /        │    │
                │   │ nonequilibrium_density_  │    │
                │   │ split                    │    │
                │   └──────────┬───────────────┘    │
                │              │ n_e                │
                │   ┌──────────▼───────────────┐    │
                │   │ poisson_solve (callback) │    │
                │   └──────────┬───────────────┘    │
                │              │ V_out              │
                │   ┌──────────▼───────────────┐    │
                │   │ mixing.LinearMixer /     │    │
                │   │         AndersonMixer    │    │
                │   └──────────┬───────────────┘    │
                │              │ V_new              │
                │   ┌──────────▼───────────────┐    │
                │   │ make_system(V_new)       │    │
                │   └──────────┬───────────────┘    │
                │              │  (next iter)       │
                └──────────────┴────────────────────┘
                                 │
                                 ▼
                          SCResult(V, n, ...)
```

## Modules

### `scpoisson/density.py`

Everything related to extracting the NEGF electron density from a
finalized kwant system.

**Low-level plumbing (private):**

- `_hamiltonian_sparse(syst, params)` — `syst.hamiltonian_submatrix` as a
  CSC sparse matrix.
- `_site_orbital_range(syst, site_idx)` — returns the
  `(start_orb, stop_orb)` slice for a given site index. Handles kwant's
  block-format `syst.site_ranges`.
- `_interface_orbitals(syst, lead_idx)` — flat array of orbital indices
  on the interface of a given lead, in the order kwant expects in its
  self-energy matrix.
- `_embed_lead_selfenergies(syst, energy, params)` — sums each lead's
  self-energy into a single N×N sparse matrix over the full orbital set.

**Green's function:**

- `retarded_gf_diagonal(syst, energy, params, eta)` — returns the
  diagonal of `G^R(z) = (zI − H − Σ_total(z))^{-1}` for each orbital.
  Accepts real or complex energy. Implemented as `splu` plus column-by-
  column solves; this is the O(N²) hot loop on bigger systems and is the
  natural place to swap in selected inversion or recursive Green's
  functions.

**LDOS and partial spectral function:**

- `ldos_at_energy(syst, energy, ...)` — `−(1/π) Im diag(G^R)`. Matches
  `kwant.ldos` to machine precision but works at complex energies.
- `partial_spectral_function(syst, energy, lead, ...)` — per-lead
  spectral function `ρ_α(i,E) = (1/2π) Σ_n |ψ_n^α(i,E)|²` using
  `kwant.wave_function`. Real-axis only.

**Fermi function:** `fermi(E, mu, kT)` — overflow-safe, handles kT=0.

**Density routines (high-level):**

- `equilibrium_density_real_axis(syst, mu, kT, e_min, e_max, n_energy,
  ...)` — trapezoid on real-energy mesh. Simple, biased near band edges.
- `equilibrium_density_contour(syst, mu, kT, e_min, n_arc, n_line, ...)` —
  Gauss-Legendre on a semicircular contour from `e_min` to µ in the
  upper half plane. **Workhorse for production.** Validated to machine
  precision against the analytic 1D-chain filling. Finite-T handled by a
  real-axis Fermi-window correction.
- `nonequilibrium_density_real_axis(syst, mu_per_lead, kT, ...)` — direct
  Σ_α ∫dE f_α ρ_α via `partial_spectral_function`. Use if you don't trust
  the split.
- `nonequilibrium_density_split(syst, mu_per_lead, kT, mu_ref, ...)` —
  Brandbyge-style equilibrium reference + real-axis bias-window
  correction. Avoids the kwant 1.5.0 band-edge bug; preferred over the
  pure real-axis form.

**Conventions:**

- All density values are in **electron number per orbital** (not charge,
  not /Å³).
- `spin_factor=2` by default (spin-1/2 electrons described by a spinless
  Hamiltonian); set to 1 if your kwant Hamiltonian already includes
  spin explicitly or for spinless toys.

### `scpoisson/poisson.py`

Finite-difference Poisson solvers, flux-conservative for non-uniform ε.

- `solve_1d(rho, dx, eps_r, bc_left, bc_right, eps0, charge_e)` — SI-unit
  solver. Stub-quality; you probably want `solve_1d_natural`.
- `solve_1d_natural(rho_e, dx, eps_r, bc_left, bc_right, coupling)` —
  natural-units (`−V'' = coupling·rho_e`). Used everywhere internally.
- `solve_2d_natural(rho, dx, dy, eps_r, bc_x_left, bc_x_right,
  bc_y_bot, bc_y_top, coupling)` — 5-point FD on `(Nx, Ny)` rectangular
  grid. Detects the all-Neumann null space and pins V[0,0]=0.

BCs are tuples `("dirichlet", value)` or `("neumann", value)`. Neumann
sign convention: `(V_next - V_edge) / d = value` at every edge (zero for
no-flux).

### `scpoisson/mixing.py`

Mixing schemes for the SC iteration. All have the same interface:
`mixer.step(V_in, V_out) -> V_new`.

- `LinearMixer(alpha)`: simple convex blend.
- `AndersonMixer(alpha, history)`: Pulay-style least-squares mixing on
  the residuals from the past `history` iterations.

### `scpoisson/sc.py`

The orchestrator.

- `SCResult` (dataclass): `V`, `n`, `iterations`, `converged`,
  `residual_history`.
- `run_sc_loop(...)`: the loop. Picks
  `equilibrium_density_contour` if all `mu_per_lead` are equal, else
  `nonequilibrium_density_split`. The Poisson step is delegated entirely
  to the user-supplied callable.
- `poisson_solve_1d(...)` / `poisson_solve_2d(...)`: factory functions
  that return ready-to-use Poisson callables for the built-in cases.
  The 2D one takes an `orbital_grid[Nx, Ny]` int array mapping each
  cell to a flat orbital index.

## Coordinate / index conventions

| Layer            | Index space                                  |
|------------------|----------------------------------------------|
| kwant            | `site_idx` (int), each site has ≥1 orbitals |
| density module   | flat `orb_idx ∈ [0, N_orb)`                  |
| Poisson 1D       | flat `i ∈ [0, N)`, one cell per orbital      |
| Poisson 2D       | `(x, y) ∈ [0,Nx) × [0,Ny)`, mapped to flat   |

The orbital-to-grid map is **user-provided** via the `orbital_grid`
argument to `poisson_solve_2d`. Build it by iterating
`enumerate(syst.sites)` and reading `site.tag`.

The flat ordering of `n_orbitals` is the same throughout — whatever order
`syst.hamiltonian_submatrix` puts orbitals in.

## Sign convention

Throughout this codebase:

- `V` (returned by the Poisson backend, stored in `res.V`, added to
  on-site by `make_system`) is the **electron on-site potential energy**,
  i.e. `V_onsite = −eφ` where φ is the electrostatic potential. Units
  match the kwant Hamiltonian (typically t).
- `n` (returned by density routines, stored in `res.n`, given as
  `n_background`) is the **electron number per orbital**. Positive.
- The Poisson source inside `run_sc_loop` is `n_e − n_background`
  (electrons minus ions, in electron-number units).

If your code is screening with the wrong sign, look here first.

## Extending the package

**A new Poisson geometry** (3D, cylindrical, half-space, …):
write a callable `f(src_flat) -> V_flat` and pass it as `poisson_solve`
to `run_sc_loop`. Build it however you like; the SC loop doesn't care.

**A new mixing scheme:** any object with a `step(V_in, V_out) -> V_new`
method works.

**A faster Green's-function diagonal:** replace
`density.retarded_gf_diagonal` with a selected-inversion implementation
or a recursive-Green's-function routine. The function signature is the
only contract.

**A parametric kwant system** (avoid rebuilding the Builder each iter):
build the kwant Builder once outside the SC loop and have `make_system(V)`
return a *finalized* system with `params=dict(V_onsite=V)`. The
Hamiltonian function on each site reads `V_onsite[i]` from params. We
haven't done this yet because the small-system MVP doesn't need it.

## What's missing

- A clean way to attach gates with non-trivial geometry beyond
  rectangular Dirichlet BCs.
- Finite-T Matsubara-pole handling in the equilibrium contour.
- Selected inversion for `retarded_gf_diagonal`. Currently O(N²) per
  energy point in solve cost.
- Recursive Green's function for very long ribbons.
- True 3D Poisson for systems with gates outside the 2D plane.
