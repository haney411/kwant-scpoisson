# User guide

How to run self-consistent NEGF + Poisson calculations with the
`scpoisson` package. Written so a physicist who knows Kwant can pick this
up cold; it is also the primer Claude reads when returning to this codebase.

## Prerequisites

```bash
pip install kwant matplotlib pytest    # already done on this machine
```

scipy ≥ 1.14 is required (the package uses sparse routines that work with
either old or new scipy, but a few of our tests assume the newer API).

## Anatomy of a calculation

Every SC-Poisson run has **four user-supplied pieces**:

1. A **system factory** `make_system(V_flat) -> kwant.system.FiniteSystem`.
   It builds the kwant Hamiltonian with the on-site shifts `V_flat[orb]`
   already in place.
2. A **Poisson backend** — any callable `poisson_solve(src_flat) ->
   V_flat`. Use `sc.poisson_solve_1d` or `sc.poisson_solve_2d` for the
   built-in choices.
3. A **background charge** `n_background[orb]` (electron-number units).
4. **Boundary data**: chemical potentials per lead `mu_per_lead`,
   temperature `kT`, and the BCs you baked into the Poisson backend.

The SC loop then converges `V_flat` such that the kwant electron density
and the Poisson potential are mutually consistent.

## Minimal 1D example

```python
import numpy as np, kwant
from scpoisson import sc, density, mixing

# 1. System factory: 1D tight-binding chain with leads.
N, t = 30, 1.0
lat = kwant.lattice.chain(norbs=1)
sym = kwant.TranslationalSymmetry((-1,))

def factory(V_flat):
    syst = kwant.Builder()
    for i in range(N):
        syst[lat(i)] = float(V_flat[i])
    for i in range(N - 1):
        syst[lat(i), lat(i+1)] = -t
    lead = kwant.Builder(sym)
    lead[lat(0)] = 0.0
    lead[lat(0), lat(1)] = -t
    syst.attach_lead(lead); syst.attach_lead(lead.reversed())
    return syst.finalized()

# 2. Reference (n_e at V=0) — usually the safest choice for n_background.
n_ref = density.equilibrium_density_contour(
    factory(np.zeros(N)), mu=0.0, kT=0.0,
    e_min=-3.0, n_arc=60, spin_factor=1,
)

# 3. Add an actual doping perturbation (positive bg in middle 4 sites).
delta = np.zeros(N); delta[N//2-2:N//2+2] = 0.01
n_bg = n_ref + delta

# 4. Pick the Poisson backend.
poisson_solve = sc.poisson_solve_1d(
    dx=1.0, coupling=8.0/N**2,                  # natural coupling scale
    bc_left=("dirichlet", 0.0),
    bc_right=("dirichlet", 0.0),
)

# 5. Run.
res = sc.run_sc_loop(
    make_system=factory, n_orbitals=N,
    poisson_solve=poisson_solve,
    mu_per_lead=[0.0, 0.0], kT=0.0,
    n_background=n_bg,
    mixer=mixing.AndersonMixer(alpha=0.5, history=5),
    tol=1e-6, max_iter=60,
    density_kw=dict(e_min=-3.0, n_arc=60),
    spin_factor=1,                              # 1 = spinless, 2 = spin-1/2
)
print(res.V, res.n, res.iterations, res.converged)
```

## Quasi-2D ribbon with full 2D Poisson (edge physics)

```python
import numpy as np, kwant
from scpoisson import sc, density, mixing

Nx, Ny, t = 14, 11, 1.0
lat = kwant.lattice.square(norbs=1)
sym = kwant.TranslationalSymmetry((-1, 0))

def factory(V_flat):
    syst = kwant.Builder()
    for x in range(Nx):
        for y in range(Ny):
            k = x*Ny + y
            syst[lat(x, y)] = 4.0*t + float(V_flat[k])     # band centered at +4t
    for x in range(Nx):
        for y in range(Ny - 1):
            syst[lat(x, y), lat(x, y+1)] = -t
    for x in range(Nx - 1):
        for y in range(Ny):
            syst[lat(x, y), lat(x+1, y)] = -t
    lead = kwant.Builder(sym)
    for y in range(Ny):
        lead[lat(0, y)] = 4.0*t
    for y in range(Ny - 1):
        lead[lat(0, y), lat(0, y+1)] = -t
    for y in range(Ny):
        lead[lat(0, y), lat(1, y)] = -t
    syst.attach_lead(lead); syst.attach_lead(lead.reversed())
    return syst.finalized()

# Build the (x, y) -> orbital map (needed by poisson_solve_2d).
syst0 = factory(np.zeros(Nx*Ny))
grid = np.full((Nx, Ny), -1, dtype=int)
for orb_idx, site in enumerate(syst0.sites):
    x, y = site.tag
    grid[x, y] = orb_idx

poisson_solve = sc.poisson_solve_2d(
    grid_shape=(Nx, Ny), orbital_grid=grid,
    dx=1.0, dy=1.0, coupling=8.0/(Nx*Ny),
    bc_x_left=("dirichlet", 0.0),  bc_x_right=("dirichlet", 0.0),
    bc_y_bot=("neumann", 0.0),     bc_y_top=("neumann", 0.0),   # free edges
)

n_ref = density.equilibrium_density_contour(
    syst0, mu=2.0, e_min=-1.0, n_arc=80, spin_factor=1,
)

# Jellium background at bulk-mean density: edges will band-bend to screen
# their density deficit.
n_bulk = n_ref.reshape(-1, Ny)[Nx//4:-Nx//4, Ny//2].mean()
n_bg = np.full(Nx*Ny, n_bulk)

res = sc.run_sc_loop(
    make_system=factory, n_orbitals=Nx*Ny,
    poisson_solve=poisson_solve,
    mu_per_lead=[2.0, 2.0], n_background=n_bg,
    mixer=mixing.AndersonMixer(),
    tol=1e-5, max_iter=40,
    density_kw=dict(e_min=-1.0, n_arc=80), spin_factor=1,
)

V_2d = res.V[grid]    # reshape flat V back to (Nx, Ny)
n_2d = res.n[grid]
```

## Non-equilibrium (finite bias)

Pass different `mu_per_lead` to enable the non-equilibrium density path
(`nonequilibrium_density_split` under the hood). For finite temperature
also pass `kT`.

```python
res = sc.run_sc_loop(
    ...,
    mu_per_lead=[+0.1, -0.1],         # bias of 0.2t between leads
    kT=0.01,
    density_kw=dict(eq_e_min=-3.0, eq_n_arc=60, bias_n_energy=80),
    ...
)
```

The non-eq path: equilibrium contour at `min(µ_L, µ_R)` (or pass
`mu_ref=` in `density_kw`) plus a real-axis bias-window correction. The
bias window must lie inside the band of each lead, or the kwant 1.5.0
mode-finder bug bites; for typical biases (\|eV\| < bandwidth) this is
not a concern.

## Boundary conditions guide

| Edge type                   | Typical BC                       |
|-----------------------------|----------------------------------|
| Longitudinal (lead end)     | `dirichlet`, V=0 (lead reference) |
| Transverse, free / vacuum   | `neumann`, ∂V/∂n=0               |
| Transverse, adjacent gate   | `dirichlet`, V=V_gate            |
| Symmetry plane              | `neumann`, ∂V/∂n=0               |

In the all-Neumann limit, the Poisson equation has a one-dimensional null
space (V is defined up to an additive constant). The solver detects this
case and pins V[0,0]=0 to remove the gauge.

## Picking `coupling`

The natural-units Poisson is `−∇²V = coupling·src` with V in the same
energy units as the Hamiltonian (e.g. t). For a region of linear size
`L` and source amplitude `s`, V scales as `coupling·s·L²/(2π²)` or so.
A useful rule of thumb that we use throughout the examples:

- 1D chain of length N: `coupling ≈ 8/N²`
- Quasi-2D ribbon `Nx × Ny`: `coupling ≈ 8/(Nx·Ny)`

Too large and V exits the band, pushing density to 0. Too small and
screening is negligible. Adjust to taste.

## Picking the energy mesh / contour

`density_kw` is forwarded to either `equilibrium_density_contour` or
`nonequilibrium_density_split`. Useful knobs:

- `e_min` (or `eq_e_min`): below the band bottom. Set conservatively.
- `n_arc` (or `eq_n_arc`): Gauss-Legendre points on the contour. 40–80
  is normally fine; the contour integrand is smooth so doubling rarely
  changes results.
- `bias_n_energy`: real-axis points in the bias window. 60–120 is fine
  for biases up to ~1 in natural units.

## Mixing schemes

- `mixing.LinearMixer(alpha)`: V_new = (1−α)·V_in + α·V_out. Stable,
  slow. α ∈ [0.1, 0.5] depending on problem stiffness.
- `mixing.AndersonMixer(alpha=0.3, history=5)`: Pulay-style. Default for
  production work; usually 2× faster than linear mixing.

## Output

`res = sc.run_sc_loop(...)` returns an `SCResult` with:

- `res.V` — converged on-site potential, shape `(n_orbitals,)`
- `res.n` — converged electron density, shape `(n_orbitals,)`
- `res.iterations` — number of SC iterations run
- `res.converged` — bool
- `res.residual_history` — `max|ΔV|` at each iteration

To reshape back to a (Nx, Ny) grid for the 2D ribbon case, index with
your orbital map: `res.V[grid]`.

## Running the tests

```bash
cd ~/wrk/kwant
python3 tests/run_all.py
```

You should see "21 passed, 0 failed".

## Common gotchas

- **MUMPS not installed** — kwant prints a noisy `RuntimeWarning` at
  import. The SciPy fallback works fine for small/medium systems; install
  `python-mumps` if you need to scale up.
- **Band edges** — kwant 1.5.0 throws at out-of-band energies in
  `kwant.smatrix`/`kwant.wave_function`. The equilibrium contour path
  avoids this. For non-eq density, keep your bias window inside the band.
- **Sign mistakes in Poisson source** — remember `V_onsite = −eφ`. The
  SC loop computes the source as `n_e − n_background` (not the reverse).
  If your screening is backwards, you have a sign error somewhere
  upstream (e.g. you flipped the meaning of `n_background`).
- **`make_system` is called once per iteration** — currently this
  rebuilds the kwant Builder from scratch. Fine for `n_orbitals` up to a
  few hundred. For larger systems, build the Builder once and use kwant's
  `params` interface to update on-site values without rebuilding.
