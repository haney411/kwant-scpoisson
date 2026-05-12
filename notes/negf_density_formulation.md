# Non-equilibrium charge density from Kwant

## NEGF formulation

For a scattering region coupled to leads α at chemical potential µ_α and
temperature T (Fermi function f_α(E) = 1/(1+exp((E-µ_α)/k_BT))), the
non-equilibrium electron density per site i is

  n_i = Σ_α ∫ dE  f_α(E)  ρ_α(i, E)             (1)

where ρ_α(i,E) is the **partial spectral function** at site i, i.e. the
contribution to the local density of states from scattering states injected
from lead α:

  ρ_α(i, E) = (1/2π) Σ_n |ψ_n^α(i; E)|²        (2)

with ψ_n^α the scattering wave function in the scattering region due to
incoming mode n in lead α, at energy E.

The sum over leads gives the total LDOS:

  ρ_tot(i,E) = Σ_α ρ_α(i,E)                     (3)

In equilibrium (all µ_α equal to µ), (1) reduces to

  n_i^eq = ∫ dE f(E) ρ_tot(i,E).                 (4)

## Mapping to Kwant APIs

| Quantity                          | Kwant call                              |
|-----------------------------------|-----------------------------------------|
| ψ_n^α(i,E) for all modes n        | `kwant.wave_function(syst, E)(alpha)`   |
| ρ_α(i,E) per site                 | `Density(syst)(ψ_n^α).sum(modes)/(2π)`  |
| ρ_tot(i,E) (sum over leads)       | `kwant.ldos(syst, E)`                   |

Kwant's `kwant.wave_function` velocity-normalizes the scattering states so
that (2) holds with the prefactor (1/2π).  Equivalently `kwant.ldos`
returns the sum (3) directly.

## Energy integration

For SC-Poisson we need (1) on every iteration. Two regimes:

### Equilibrium part (all leads at common reference µ_ref)
Use complex-energy contour integration: a semicircle in the upper half plane
from E_min (below the band) to µ_ref. Integrand is analytic above the real
axis, so few quadrature points suffice (typically 20-50 Gauss-Legendre
points on a circular arc). Compute `G^R(E+iη)` directly.

This avoids the kwant 1.5.0 mode-finder bug at band edges (see
notes/kwant_lead_bug.md) and is the standard approach (Brandbyge et al.
2002, Areshkin & Nikolic 2010).

### Non-equilibrium correction (bias window)
For lead bias µ_L ≠ µ_R, decompose density as

  n_i = n_i^eq(µ_ref) + Σ_α ∫_{µ_ref}^{µ_α} dE ρ_α(i,E) [f_α(E) - f_ref(E)]

The second term is integrated along the real axis only within the bias
window |E - µ_ref| ≲ |eV| + few k_BT — a narrow range where
`kwant.wave_function` is needed. Per-lead spectral functions are required
here (LDOS won't do).

Reference choice µ_ref: pick µ_ref = (µ_L + µ_R)/2 to minimize the real-axis
piece, or pick the lead with the larger band overlap. Convention is
"equilibrium with the lead at lowest chemical potential, then add
non-equilibrium correction from higher-potential leads" (Brandbyge).

## Conventions in this project

- Energy units: same as Kwant Hamiltonian (we pass `t` as the energy scale).
- Density units: dimensionless (electrons per site/orbital).
- Sign: n_i is the **electron** density (positive number). The charge
  density that drives Poisson is ρ(r) = e[n_doping(r) − n_i] for n-type;
  details handled in `poisson.py`.
- Spin: factor of 2 for spin degeneracy is applied at the integration step,
  controlled by `spin_factor` parameter (default 2 = spinless Kwant
  Hamiltonian for spinful electrons).

## References

- S. Datta, "Quantum Transport: Atom to Transistor" (CUP 2005), ch. 8.
- M. Brandbyge, J.-L. Mozos, P. Ordejón, J. Taylor, K. Stokbro,
  "Density-functional method for nonequilibrium electron transport,"
  Phys. Rev. B 65, 165401 (2002). — Eq. 16-22 for the eq+neq density split.
- D. A. Areshkin, B. K. Nikolić, "Electron density and transport in
  top-gated graphene nanoribbon devices: First-principles Green function
  algorithms for systems containing a large number of atoms,"
  Phys. Rev. B 81, 155450 (2010).
- C. W. Groth, M. Wimmer, A. R. Akhmerov, X. Waintal, "Kwant: a software
  package for quantum transport," New J. Phys. 16, 063065 (2014).
