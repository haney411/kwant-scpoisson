# scpoisson — Self-Consistent Poisson for Kwant

Self-consistent NEGF + Poisson solver bolted onto [Kwant](https://kwant-project.org).
1D longitudinal transport, scaling to quasi-2D (finite transverse extent,
non-periodic).

## Status

MVP works end-to-end: install, density extraction, Poisson, SC loop for
both equilibrium and non-equilibrium (finite bias), 1D chain and quasi-2D
ribbon. 15 tests, all passing.

## Layout

```
scpoisson/
  density.py     NEGF density: real-axis LDOS, complex contour, per-lead
                 spectral function, equilibrium + non-eq decomposition
  poisson.py     1D and 2D finite-difference Poisson with Dirichlet/Neumann BCs
  mixing.py      Linear and Anderson mixing for SC iteration
  sc.py          Top-level SC loop orchestrator + poisson_solve_1d/2d factories
notes/           Physics formulation; kwant compatibility notes
examples/        End-to-end demos (1D chain, ribbon)
tests/           Unit + integration tests
```

## Quick start

```python
import numpy as np
import kwant
from scpoisson import sc, density, mixing

def factory(V_onsite):
    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    for i in range(N):
        syst[lat(i)] = float(V_onsite[i])
    for i in range(N - 1):
        syst[lat(i), lat(i+1)] = -1.0
    sym = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym); lead[lat(0)] = 0.0; lead[lat(0), lat(1)] = -1.0
    syst.attach_lead(lead); syst.attach_lead(lead.reversed())
    return syst.finalized()

# Set up the Poisson backend (1D here; use sc.poisson_solve_2d for ribbons):
poisson_solve = sc.poisson_solve_1d(
    dx=1.0, coupling=8.0 / N**2,
    bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
)

# Run SC loop:
res = sc.run_sc_loop(
    make_system=factory, n_orbitals=N,
    poisson_solve=poisson_solve,
    mu_per_lead=[0.0, 0.0],
    n_background=n_bg,
    mixer=mixing.AndersonMixer(),
    tol=1e-6, max_iter=60,
)
print(res.V)   # self-consistent on-site potential
print(res.n)   # self-consistent electron density
```

For a quasi-2D ribbon with finite transverse extent (edge effects),
use the 2D Poisson backend with an orbital map:

```python
poisson_solve = sc.poisson_solve_2d(
    grid_shape=(Nx, Ny), orbital_grid=grid,   # (x,y) -> orbital index
    dx=1.0, dy=1.0, coupling=8.0/(Nx*Ny),
    bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
    bc_y_bot=("neumann", 0.0), bc_y_top=("neumann", 0.0),    # free edges
)
```

See `examples/example_sc_ribbon_2d.py` for a runnable demo that contrasts
2D Poisson (resolves edge band-bending) vs transverse-averaged 1D.

## Run the tests

```bash
python3 tests/run_all.py
```

## Key technical choices

- **Equilibrium density via complex-contour Gauss-Legendre on a semicircle
  in the upper half-plane.** Machine-precision agreement with the analytic
  filling of a uniform 1D chain (test).
- **Non-equilibrium density via split: equilibrium reference + real-axis
  bias-window correction** using per-lead spectral functions from
  `kwant.wave_function`. Brandbyge et al. (2002) style.
- **Anderson mixing** for SC iteration — ~2× faster than linear on
  the small problems we've tested.
- **kwant 1.5.0 mode-finder bug** (scipy ≥1.14 incompatibility at
  out-of-band energies) is sidestepped by the contour-integration approach.
  See `notes/kwant_lead_bug.md`.

## Known limitations / next steps

- Real-axis density methods use trapezoid; biased near band edges. Contour
  method is the workhorse.
- Finite-T contour treatment is approximate (real-axis Fermi-window
  correction). Proper Matsubara-pole treatment is a follow-up.
- `make_system(V)` rebuilds the kwant Builder each iteration. For larger
  systems, switch to a parametric Hamiltonian via Kwant's `params` interface.
- LU-per-column G^R diagonal extraction is O(N²) per energy. For larger N,
  swap in selected inversion / recursive Green's function.
- Install MUMPS bindings (`pip install python-mumps`) for faster sparse
  solves on bigger ribbons.

## References

- Datta, *Quantum Transport: Atom to Transistor* (CUP 2005), ch. 8.
- Brandbyge, Mozos, Ordejón, Taylor, Stokbro, "Density-functional method
  for nonequilibrium electron transport," PRB **65**, 165401 (2002).
- Areshkin, Nikolić, "Electron density and transport in top-gated graphene
  nanoribbon devices..." PRB **81**, 155450 (2010).
- Groth, Wimmer, Akhmerov, Waintal, "Kwant: a software package for quantum
  transport," NJP **16**, 063065 (2014).
