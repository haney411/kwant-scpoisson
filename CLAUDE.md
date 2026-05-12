# CLAUDE.md

Guidance for future Claude sessions working on this codebase. Keep it
tight; the detailed docs live in `docs/`.

## What this is

`scpoisson` — a self-consistent NEGF + Poisson solver built on top of
Kwant. Target: 1D longitudinal transport scaling to quasi-2D systems
(finite transverse extent, non-periodic transverse direction). User
cares specifically about **edge effects** in the electrostatics, so the
full 2D Poisson path (not transverse-averaged) is the production path.

Read `docs/PROCESS.md` for the history of how this got built and why.
Read `docs/ARCHITECTURE.md` for the module-by-module API surface.
Read `docs/USER_GUIDE.md` for how to drive the package.
Read `docs/TESTING.md` for the validation strategy and current test list.

## Run before claiming any change works

```bash
python3 tests/run_all.py
```

Expect "21 passed, 0 failed". The suite covers contour density vs analytic
1D-chain filling (machine precision), 2D Poisson vs analytic Fourier
mode, SC screening sign, edge-bending differential between 2D and 1D
Poisson, etc. **If any test fails after your change, fix it before
moving on** — the tests are not flaky.

## Conventions to not break

- **Sign convention.** `V` (Poisson output, kwant on-site shift) is the
  electron *potential energy* `V_onsite = −eφ`. The Poisson source
  inside `run_sc_loop` is `n_e − n_background`. If screening goes the
  wrong way, the bug is here.
- **Density units.** Per orbital, electron number (positive). Apply
  `spin_factor` (default 2) at the integration step; toy spinless tests
  pass `spin_factor=1` and that is *not* a bug.
- **Natural coupling scale.** 1D chain length N → `coupling ≈ 8/N²`.
  Ribbon Nx×Ny → `coupling ≈ 8/(Nx·Ny)`. Larger values push V out of
  band and expel electrons; smaller values give negligible screening.
  This isn't a hard constraint, but if you pick something an order of
  magnitude off, expect the SC loop to misbehave.
- **Convergence metric.** `max|ΔV|` (absolute max-norm), not relative.
  Relative-to-`||V||` divides by ≈0 at the start and is meaningless.

## kwant 1.5.0 bug — DO NOT touch the upstream package

`kwant.smatrix` / `kwant.wave_function` raise at out-of-band energies on
single-orbital leads with scipy ≥ 1.14 (`la.solve` on a non-square
matrix). We sidestep this by design:

- Equilibrium density goes through `equilibrium_density_contour`, which
  uses `lead.selfenergy(z)` at complex z (works fine) and never touches
  the buggy code path.
- Non-eq density splits as `equilibrium_density_contour(µ_ref) +
  bias-window real-axis correction`. The bias window must lie inside the
  lead band; for any sensible setup (`|eV| < bandwidth`) this is fine.

Don't try to "fix" the bug by patching the system Python kwant install.
Don't suggest sampling `kwant.smatrix` outside the band.

## Build before you optimize

The MVP runs everything correctly but is not fast:

- `make_system(V)` rebuilds the kwant Builder from scratch each iter.
  For modest sizes (≲ few hundred orbitals) this costs ~10–100 ms;
  cheap enough.
- `retarded_gf_diagonal` extracts the diagonal of `G^R(z)` by `splu` +
  column-by-column solves. O(N²) per energy point; the natural place
  to swap in selected inversion (`scipy.sparse.linalg` doesn't have it;
  options: `pyselinv`, recursive Green's function, or just dense LU
  + diag for N < 500).

If you're asked to scale up: do not refactor for performance speculatively.
Profile, then change one thing.

## Style notes for this repo

- No emoji.
- No multi-paragraph docstrings or top-of-file decorative banners.
  Module docstrings are short.
- Comments only when WHY is non-obvious. The sign-convention comment in
  `sc.py` is the right level; don't add running narrative.
- Tests are plain functions; no pytest fixtures needed for this size.
- New tests must assert something *physical* (analytic match, sign,
  symmetry) — not "the output stays the same as last run".

## User preferences for this collaboration

- Pick sensible defaults and proceed; don't open a multi-question
  AskUserQuestion menu at the start of a task. The user redirects when
  needed.
- This is a dedicated scientific workstation. No venv. `pip install`
  works because we set `~/.config/pip/pip.conf` to use
  `--user --break-system-packages`.
- User is a CM/electronic-structure researcher. Physics shorthand
  (NEGF, self-energy, Friedel, Brandbyge-split, gauge pin, etc.) is
  fine without unpacking.
- The user is interested in edges and edge electrostatics. Default the
  2D Poisson with Neumann transverse edges; that's the physics target.

## When extending

- A new Poisson geometry: write a callable `f(src_flat) -> V_flat`
  and pass as `poisson_solve=`. Don't add config-knob bloat to
  `run_sc_loop`.
- A new mixing scheme: object with `.step(V_in, V_out) -> V_new`.
- A new density routine: add to `density.py`, keep the
  `equilibrium_density_contour` signature compatible, write at least one
  test against either an analytic reference or one of the existing
  routines.

## Things to flag to the user explicitly

- Any time you change `sc.py`'s Poisson-source sign or scaling.
- Any time a regression test loosens its tolerance.
- Any new kwant or scipy version mismatch that breaks the existing
  workaround.

## Where to put new things

```
scpoisson/      package code
tests/          test_<thing>.py + add to run_all.py's `test_modules`
examples/       runnable demos; can import from scpoisson; not in test path
notes/          physics derivations, references, compatibility notes
docs/           PROCESS.md, USER_GUIDE.md, ARCHITECTURE.md, TESTING.md
scratch/        ephemeral; don't commit anything important here
```
