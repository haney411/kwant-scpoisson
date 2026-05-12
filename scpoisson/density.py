"""NEGF charge density for Kwant systems.

Two ways to compute density are provided:

1. **Real-axis LDOS integration** (`equilibrium_density_real_axis`,
   `nonequilibrium_density_real_axis`). Uses `kwant.ldos` /
   `kwant.wave_function` on a real-energy mesh. Simple and direct.
   Sensitive to band-edge singularities and to the kwant 1.5.0 mode-finder
   bug at out-of-band energies (see notes/kwant_lead_bug.md), so the energy
   mesh must be set carefully.

2. **Complex-contour integration** (`equilibrium_density_contour`). For the
   equilibrium piece, integrate G^R(z) along a semicircular contour in the
   upper half plane from a low energy E_min (below the band) up to the
   chemical potential. The integrand is analytic above the real axis, so
   ~30 Gauss-Legendre points on the arc converge well, and the band-edge
   bug is avoided entirely. This is the workhorse for SC-Poisson.

Sign / spin conventions
-----------------------
- Density returned is the electron number per site (or per orbital for
  multi-orbital sites). Positive.
- A `spin_factor` argument (default 2) scales the result for systems where
  the Kwant Hamiltonian is spinless but represents spin-1/2 electrons.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import kwant
from kwant.operator import Density


# --------------------------------------------------------------------- #
# Building blocks                                                        #
# --------------------------------------------------------------------- #

def _hamiltonian_sparse(syst, params=None):
    """Return the scattering-region Hamiltonian as a sparse CSC matrix."""
    H = syst.hamiltonian_submatrix(sparse=True, params=params)
    return H.tocsc()


def _site_orbital_range(syst, site_idx):
    """Return (start_orbital, stop_orbital) for the given site index.

    syst.site_ranges has the form
        [(first_site_in_block, norbs_per_site, first_orb_in_block), ...]
    where the last tuple is a sentinel: (n_sites, 0, n_total_orbitals).
    """
    sr = syst.site_ranges
    for k in range(len(sr) - 1):
        first_site, norbs, first_orb = sr[k]
        next_first_site = sr[k + 1][0]
        if first_site <= site_idx < next_first_site:
            offset = (site_idx - first_site) * norbs
            return first_orb + offset, first_orb + offset + norbs
    raise IndexError(f"site index {site_idx} out of range for system")


def _interface_orbitals(syst, lead_idx):
    """Return a flat array of orbital indices for the interface of a lead."""
    interface = syst.lead_interfaces[lead_idx]
    orbs = []
    for site_idx in interface:
        a, b = _site_orbital_range(syst, site_idx)
        orbs.extend(range(a, b))
    return np.asarray(orbs, dtype=int)


def _embed_lead_selfenergies(syst, energy, params=None):
    """Return a sparse embedding of Σ^R_total(E) = Σ_α Σ_α^R(E).

    Kwant gives the lead self-energy only on the lead-interface orbitals.
    We sum-embed each lead's contribution into a single N×N matrix
    (N = total orbital count of the scattering region).
    """
    n = syst.hamiltonian_submatrix(sparse=True, params=params).shape[0]
    Sigma = sp.lil_matrix((n, n), dtype=complex)
    for lead_idx, lead in enumerate(syst.leads):
        try:
            se = lead.selfenergy(energy, params=params)
        except TypeError:
            # Some old kwant signatures may not accept params kwarg
            se = lead.selfenergy(energy)
        orbs = _interface_orbitals(syst, lead_idx)
        if se.shape[0] != orbs.size:
            raise RuntimeError(
                f"lead {lead_idx}: selfenergy shape {se.shape} does not "
                f"match interface orbital count {orbs.size}"
            )
        # Block-add
        for a in range(orbs.size):
            for b in range(orbs.size):
                Sigma[orbs[a], orbs[b]] += se[a, b]
    return Sigma.tocsc()


def retarded_gf_diagonal(syst, energy, params=None, eta=0.0):
    """Compute the diagonal of G^R(E) for the scattering region.

    G^R(E) = [(E + iη) I − H − Σ_total(E)]^{-1}

    Parameters
    ----------
    syst : finalized kwant system
    energy : float or complex
        If complex, η is ignored. If real, ``eta`` adds a small
        positive imaginary part for numerical stability.
    params : dict or None
        Parameter dictionary for the Hamiltonian.
    eta : float
        Small broadening for real-energy calls. Default 0.0 because the
        lead self-energies already supply the broadening for in-band E.

    Returns
    -------
    diag_GR : (N,) complex array
        Diagonal entries of G^R, one per orbital in the scattering region.

    Notes
    -----
    For O(N) inversion this would use a recursive Green's function or
    selected-inversion scheme. For the moderate sizes (≲ few thousand
    orbitals) of our first targets we just do a direct LU solve column by
    column on the identity. For larger systems we'll swap in a smarter
    backend.
    """
    z = energy + 1j * eta if np.isrealobj(energy) else energy
    H = _hamiltonian_sparse(syst, params=params)
    Sigma = _embed_lead_selfenergies(syst, energy, params=params)
    N = H.shape[0]
    A = sp.eye(N, format="csc") * z - H - Sigma
    # Solve A X = I, but we only need diag(X). The cheapest correct way
    # without a selected-inversion library is to factorize and solve column
    # by column, taking diag entries. For N up to a few thousand this is
    # fine; for larger N switch to mumps / selected inversion.
    LU = spla.splu(A.tocsc())
    diag = np.empty(N, dtype=complex)
    e = np.zeros(N, dtype=complex)
    for i in range(N):
        e[i] = 1.0
        x = LU.solve(e)
        diag[i] = x[i]
        e[i] = 0.0
    return diag


def ldos_at_energy(syst, energy, params=None, eta=0.0):
    """Total LDOS per orbital at a given (possibly complex) energy.

    LDOS_i(E) = − (1/π) Im G^R_ii(E + iη)
    """
    diag = retarded_gf_diagonal(syst, energy, params=params, eta=eta)
    return -(1.0 / np.pi) * diag.imag


def partial_spectral_function(syst, energy, lead, params=None):
    """Per-lead partial spectral function ρ_α(i, E) using wave functions.

    Uses kwant.wave_function — only valid at real energies inside the band
    of lead α. Returns a (n_sites_or_orbitals,) array.
    """
    wf = kwant.wave_function(syst, energy, params=params)
    psi = wf(lead)  # shape (n_modes, n_orbitals)
    if psi.shape[0] == 0:
        N = syst.hamiltonian_submatrix(sparse=True, params=params).shape[0]
        return np.zeros(N)
    # |ψ|² per orbital, summed over modes, divided by 2π.
    return np.sum(np.abs(psi) ** 2, axis=0) / (2.0 * np.pi)


# --------------------------------------------------------------------- #
# Fermi function                                                         #
# --------------------------------------------------------------------- #

def fermi(E, mu, kT):
    """Fermi-Dirac, robust at kT=0."""
    if kT <= 0:
        return np.where(np.real(E) < mu, 1.0, 0.0)
    x = (np.real(E) - mu) / kT
    out = np.empty_like(x, dtype=float)
    # avoid overflow
    pos = x > 0
    out[pos] = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


# --------------------------------------------------------------------- #
# Equilibrium density — real-axis integration (simple)                   #
# --------------------------------------------------------------------- #

def equilibrium_density_real_axis(syst, mu, kT=0.0, n_energy=400,
                                  e_min=None, e_max=None, params=None,
                                  spin_factor=2, eta=0.0):
    """Equilibrium electron density via real-axis LDOS integration.

        n_i = spin_factor · ∫ dE f(E−µ; kT) LDOS_i(E)

    The integration mesh runs from e_min to e_max with `n_energy`
    trapezoidal points. e_min should sit below the band bottom; e_max
    sufficiently above µ that the Fermi factor is negligible (a few k_BT
    plus a small buffer).

    Use `equilibrium_density_contour` for production work — this routine
    is mainly for sanity checking and small systems.
    """
    if e_min is None:
        e_min = mu - 20.0  # very conservative
    if e_max is None:
        e_max = mu + max(20.0 * kT, 1e-3)  # at T=0 the Fermi factor cuts at mu

    # We integrate up to E = mu (sharp Fermi at T=0) and add a small tail at T>0.
    if kT <= 0:
        # Pure step: integrate LDOS from e_min to mu
        energies = np.linspace(e_min, mu, n_energy)
        # nudge slightly below mu to be safe at exact band edges; minor.
    else:
        energies = np.linspace(e_min, mu + 15.0 * kT, n_energy)

    dE = energies[1] - energies[0]
    N = syst.hamiltonian_submatrix(sparse=True, params=params).shape[0]
    n = np.zeros(N)
    for E in energies:
        ldos_E = ldos_at_energy(syst, E, params=params, eta=eta)
        f = fermi(np.array([E]), mu, kT)[0] if kT > 0 else (1.0 if E < mu else 0.0)
        n += f * ldos_E * dE
    return spin_factor * n


# --------------------------------------------------------------------- #
# Equilibrium density — complex contour integration                      #
# --------------------------------------------------------------------- #

def _gauss_legendre(n, a, b):
    """Gauss-Legendre nodes/weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def equilibrium_density_contour(syst, mu, kT=0.0, e_min=None,
                                n_arc=40, n_line=20,
                                params=None, spin_factor=2):
    """Equilibrium electron density via complex contour integration.

    Closed contour in upper half plane: semicircle from E_min to µ (T=0)
    or from E_min to µ + few k_BT (T>0). The integrand
        f(z) G^R(z)
    is analytic in the UHP (poles of f are at Matsubara frequencies for
    T>0, treated via residues; for T=0 the contour just terminates at µ).

    For T=0 the formula is:

        n_i = -spin_factor · (1/π) Im ∮_C dz G^R_ii(z)

    where C goes from E_min along a semicircle in the UHP up to µ.

    For finite T, the rigorous treatment includes Matsubara poles; for now
    we implement T=0 (and approximate finite T by an extended contour +
    real-axis tail through the Fermi window). MVP — production-grade
    Matsubara handling is a follow-up.

    Returns n_i per orbital.
    """
    if e_min is None:
        e_min = mu - 10.0
    if e_min >= mu:
        raise ValueError(f"e_min ({e_min}) must be below mu ({mu})")

    # Semicircle in UHP from e_min to mu, radius R = (mu - e_min)/2,
    # parametrized by angle θ ∈ (π, 0) so z = center + R e^{iθ}.
    center = 0.5 * (mu + e_min)
    R = 0.5 * (mu - e_min)
    thetas, wts = _gauss_legendre(n_arc, np.pi, 0.0)
    # z(θ) = center + R e^{iθ},  dz = i R e^{iθ} dθ

    N = syst.hamiltonian_submatrix(sparse=True, params=params).shape[0]
    integral = np.zeros(N, dtype=complex)
    for theta, w in zip(thetas, wts):
        z = center + R * np.exp(1j * theta)
        dz_dtheta = 1j * R * np.exp(1j * theta)
        diag = retarded_gf_diagonal(syst, z, params=params)
        # No Fermi factor at T=0; the contour itself terminates at µ.
        integral += w * dz_dtheta * diag

    # n_i = -(1/π) Im ∮ G^R dz  (× spin factor)
    n = -(1.0 / np.pi) * integral.imag

    if kT > 0:
        # Real-axis correction in the Fermi window [µ − few kT, µ + few kT]:
        # ∫_µ^∞ f(E)·LDOS(E) dE − ∫_{-∞}^µ (1-f(E))·LDOS(E) dE
        # Approximate by symmetric window:
        Ewin = 15.0 * kT
        en, we = _gauss_legendre(n_line, mu - Ewin, mu + Ewin)
        for E, w in zip(en, we):
            f = fermi(np.array([E]), mu, kT)[0]
            # The contour integration up to µ already gave half-step at E=µ;
            # remove the step assumption and add proper Fermi smearing.
            ldos_E = ldos_at_energy(syst, E)
            n += w * (f - (1.0 if E < mu else 0.0)) * ldos_E

    return spin_factor * n


# --------------------------------------------------------------------- #
# Non-equilibrium density                                                #
# --------------------------------------------------------------------- #

def nonequilibrium_density_real_axis(
    syst,
    mu_per_lead,
    kT=0.0,
    e_min=None,
    e_max=None,
    n_energy=400,
    params=None,
    spin_factor=2,
):
    """Non-equilibrium density via per-lead spectral functions on real axis.

        n_i = spin_factor · Σ_α ∫ dE f_α(E) ρ_α(i, E)

    where ρ_α is from `partial_spectral_function`. Stays inside the band
    of each lead to avoid the kwant 1.5.0 mode-finder bug.

    Parameters
    ----------
    mu_per_lead : sequence of float
        Chemical potential of each lead, length = number of leads.
    e_min, e_max : float, optional
        Energy mesh bounds. Default to span the union of bias windows.

    Notes
    -----
    For large bias windows or wide bands, prefer
    `nonequilibrium_density_split` which uses equilibrium contour for the
    bulk part and a real-axis correction only in the bias window.
    """
    mu_per_lead = np.asarray(mu_per_lead, dtype=float)
    if e_min is None:
        e_min = mu_per_lead.min() - max(20.0 * kT, 5.0)
    if e_max is None:
        e_max = mu_per_lead.max() + max(20.0 * kT, 5.0)

    energies = np.linspace(e_min, e_max, n_energy)
    dE = energies[1] - energies[0]

    N = syst.hamiltonian_submatrix(sparse=True, params=params).shape[0]
    n = np.zeros(N)
    for E in energies:
        for alpha, mu_a in enumerate(mu_per_lead):
            f_a = fermi(np.array([E]), mu_a, kT)[0] if kT > 0 else (1.0 if E < mu_a else 0.0)
            if f_a == 0.0:
                continue
            rho_a = partial_spectral_function(syst, E, alpha, params=params)
            n += f_a * rho_a * dE
    return spin_factor * n


def nonequilibrium_density_split(
    syst,
    mu_per_lead,
    kT=0.0,
    mu_ref=None,
    eq_e_min=None,
    eq_n_arc=40,
    bias_n_energy=80,
    params=None,
    spin_factor=2,
):
    """Recommended NEGF density: equilibrium contour + bias-window real-axis.

        n_i = n_i^eq(µ_ref) + Σ_α ∫ dE [f_α(E) − f_ref(E)] ρ_α(i, E)

    where µ_ref is a reference chemical potential (default = min lead µ,
    following Brandbyge), and the second integral is needed only over the
    bias window (a few k_BT around the µ_α's).

    The equilibrium part uses `equilibrium_density_contour` (robust). The
    bias correction uses `partial_spectral_function` (real-axis, in band).
    """
    mu_per_lead = np.asarray(mu_per_lead, dtype=float)
    if mu_ref is None:
        mu_ref = mu_per_lead.min()

    # Equilibrium reference density.
    n_eq = equilibrium_density_contour(
        syst, mu=mu_ref, kT=kT, e_min=eq_e_min, n_arc=eq_n_arc,
        params=params, spin_factor=1,  # we'll apply spin factor at the end
    )

    # Bias window correction per lead.
    N = n_eq.size
    n_corr = np.zeros(N)
    e_min = min(mu_ref, mu_per_lead.min()) - max(15.0 * kT, 1e-6)
    e_max = max(mu_ref, mu_per_lead.max()) + max(15.0 * kT, 1e-6)
    if e_max > e_min:
        en, we = _gauss_legendre(bias_n_energy, e_min, e_max)
        for E, w in zip(en, we):
            f_ref = fermi(np.array([E]), mu_ref, kT)[0] if kT > 0 else (1.0 if E < mu_ref else 0.0)
            for alpha, mu_a in enumerate(mu_per_lead):
                f_a = fermi(np.array([E]), mu_a, kT)[0] if kT > 0 else (1.0 if E < mu_a else 0.0)
                df = f_a - f_ref
                if abs(df) < 1e-14:
                    continue
                rho_a = partial_spectral_function(syst, E, alpha, params=params)
                n_corr += w * df * rho_a

    return spin_factor * (n_eq + n_corr)
