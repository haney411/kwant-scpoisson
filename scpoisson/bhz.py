"""BHZ model with optional magnetic exchange (Chern / QAH systems).

Discretized on a square lattice in the basis (E↑, H↑, E↓, H↓).

Continuum Hamiltonian:
    H(k) = ε(k) I_4 + A k_x α_x + A k_y α_y + (M - B k²) β + H_exch(r)
with
    α_x = diag_block(σ_x, -σ_x),  α_y = diag_block(σ_y, σ_y),
    β   = diag(1, -1, 1, -1)  (orbital σ_z on (E,H)),
    ε(k) = C - D k²,
    H_exch(r) = m(r) · s ⊗ diag_orb(G_E, G_H),
where s = (s_x, s_y, s_z) are spin Pauli matrices acting on (↑,↓).

Topological / QAH phases (rough rules, B,D > 0 conventions):
- Without exchange (G_E = G_H = 0): QSH if sign(M)=sign(B), trivial otherwise.
- With G_E ≠ G_H and large enough m_z: one spin block inverts while the other
  doesn't → Chern insulator (C = ±1).
"""

from __future__ import annotations

import numpy as np
import kwant


# 4×4 building blocks in basis (E↑, H↑, E↓, H↓).
_I4 = np.eye(4, dtype=complex)
_BETA = np.diag([1.0, -1.0, 1.0, -1.0]).astype(complex)
_ALPHA_X = np.array(
    [[0, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, -1],
     [0, 0, -1, 0]], dtype=complex,
)
_ALPHA_Y = np.array(
    [[0, -1j, 0, 0],
     [1j, 0, 0, 0],
     [0, 0, 0, -1j],
     [0, 0, 1j, 0]], dtype=complex,
)
_SX = np.array(
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 1, 0, 0]], dtype=complex,
)
_SY = np.array(
    [[0, 0, -1j, 0],
     [0, 0, 0, -1j],
     [1j, 0, 0, 0],
     [0, 1j, 0, 0]], dtype=complex,
)
_SZ = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)


def bhz_onsite_matrix(A, B, C, D, M, a=1.0):
    """4×4 on-site BHZ matrix (no exchange, no scalar potential)."""
    return (C - 4.0*D/a**2) * _I4 + (M - 4.0*B/a**2) * _BETA


def bhz_hopx_matrix(A, B, D, a=1.0):
    """4×4 hopping matrix for the +x neighbor."""
    return (D/a**2) * _I4 + (B/a**2) * _BETA + (-1j*A/(2.0*a)) * _ALPHA_X


def bhz_hopy_matrix(A, B, D, a=1.0):
    """4×4 hopping matrix for the +y neighbor."""
    return (D/a**2) * _I4 + (B/a**2) * _BETA + (-1j*A/(2.0*a)) * _ALPHA_Y


def exchange_matrix(m, G_E=1.0, G_H=0.0):
    """4×4 on-site exchange from magnetization m = (m_x, m_y, m_z).

    Form: m·s ⊗ diag_orb(G_E, G_H). G_E, G_H are real exchange couplings to
    the E and H bands respectively. With G_E ≠ G_H, m_z gaps/inverts the
    two spin blocks asymmetrically — the mechanism for QAH.
    """
    mx, my, mz = m
    G = np.diag([G_E, G_H, G_E, G_H]).astype(complex)
    return mx * (_SX @ G) + my * (_SY @ G) + mz * (_SZ @ G)


def domain_wall_magnetization(*, axis="x", center=0.0, width=2.0, m0=1.0,
                              kind="ising"):
    """Standard 1D domain-wall profile m(r) = (m_x, m_y, m_z).

    Parameters
    ----------
    axis : "x" or "y"
        Coordinate along which the wall normal lies (m_z changes sign).
    center : float
        Position of the wall center, in lattice units.
    width : float
        Wall width w (the tanh argument is (coord - center)/w).
    m0 : float
        Saturation magnetization amplitude.
    kind : {"ising", "bloch", "neel"}
        Profile type:
        - "ising": m_z = m0·tanh(s),  m_x = m_y = 0  (simplest QAH-flip wall)
        - "bloch": m_z = m0·tanh(s),  m_y = m0·sech(s),  m_x = 0
        - "neel":  m_z = m0·tanh(s),  m_x = m0·sech(s),  m_y = 0
        where s = (coord − center)/width.

    Returns
    -------
    m_func(x, y) -> (m_x, m_y, m_z)
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if kind not in ("ising", "bloch", "neel"):
        raise ValueError("kind must be 'ising', 'bloch', or 'neel'")

    def m_func(x, y):
        coord = x if axis == "x" else y
        s = (coord - center) / width
        tz = np.tanh(s)
        sech = 1.0 / np.cosh(s)
        mz = m0 * tz
        if kind == "ising":
            return (0.0, 0.0, mz)
        if kind == "bloch":
            return (0.0, m0 * sech, mz)
        return (m0 * sech, 0.0, mz)  # neel

    return m_func


def uniform_magnetization(m=(0.0, 0.0, 1.0)):
    """Constant magnetization at every site."""
    m = tuple(float(c) for c in m)

    def m_func(x, y):
        return m

    return m_func


def make_bhz_system(
    Nx, Ny, *,
    A=1.0, B=1.0, C=0.0, D=0.0, M=1.0,
    G_E=0.0, G_H=0.0,
    magnetization=None,
    V_onsite=None,
    a=1.0,
    leads=True,
    lead_axis="x",
    lead_magnetization="auto",
):
    """Build a finalized kwant system for the BHZ model on a rectangle.

    Geometry: Nx × Ny square lattice (norbs=4). Two leads are attached along
    `lead_axis` ('x' or 'y'). Set `leads=False` to build the scattering region
    only (useful for periodic / closed-system analyses).

    Parameters
    ----------
    Nx, Ny : int
        Grid size.
    A, B, C, D, M : float
        BHZ continuum parameters. With B>0, sign(M)=sign(B) is the topological
        regime in the absence of exchange.
    G_E, G_H : float
        Exchange couplings of the magnetization to the E and H bands.
    magnetization : callable or None
        ``m(x, y) -> (m_x, m_y, m_z)`` with x, y in lattice-unit coordinates,
        or ``None`` for no exchange.
    V_onsite : array-like or None
        Optional scalar on-site potential, length Nx*Ny, applied as
        V_onsite[i]·I_4 at site i. Indexing: i = x*Ny + y.
    a : float
        Lattice constant.
    leads : bool
        Whether to attach two semi-infinite leads.
    lead_axis : "x" or "y"
        Direction along which leads extend.
    lead_magnetization : {"auto", None, callable, tuple}
        - "auto" (default): use the scattering-region magnetization evaluated
          at the lead's interface column (so the lead matches the asymptote
          there). For a domain wall along x, this naturally gives a left
          lead with m → −m0 and a right lead with m → +m0.
        - ``None``: leads are non-magnetic (pure BHZ).
        - callable ``m(y)``: per-site magnetization for both leads.
        - ``(m_left, m_right)``: explicit magnetization tuples or callables.
    """
    if lead_axis not in ("x", "y"):
        raise ValueError("lead_axis must be 'x' or 'y'")

    lat = kwant.lattice.square(a=a, norbs=4)
    syst = kwant.Builder()

    H_on = bhz_onsite_matrix(A, B, C, D, M, a=a)
    Hx = bhz_hopx_matrix(A, B, D, a=a)
    Hy = bhz_hopy_matrix(A, B, D, a=a)

    for x in range(Nx):
        for y in range(Ny):
            mat = H_on.copy()
            if magnetization is not None and (G_E != 0.0 or G_H != 0.0):
                mat = mat + exchange_matrix(
                    magnetization(x * a, y * a), G_E=G_E, G_H=G_H,
                )
            if V_onsite is not None:
                mat = mat + V_onsite[x * Ny + y] * _I4
            syst[lat(x, y)] = mat

    for x in range(Nx - 1):
        for y in range(Ny):
            syst[lat(x + 1, y), lat(x, y)] = Hx
    for x in range(Nx):
        for y in range(Ny - 1):
            syst[lat(x, y + 1), lat(x, y)] = Hy

    if leads:
        _attach_leads(
            syst, lat, Nx, Ny, a, H_on, Hx, Hy,
            magnetization=magnetization,
            G_E=G_E, G_H=G_H,
            lead_axis=lead_axis,
            lead_magnetization=lead_magnetization,
        )

    return syst.finalized()


def _resolve_lead_m(spec, magnetization, x_iface, transverse_N, a, side):
    if spec is None:
        return lambda y: None
    if spec == "auto":
        if magnetization is None:
            return lambda y: None
        return lambda y: magnetization(x_iface * a, y * a)
    if callable(spec):
        return lambda y: spec(y * a)
    if isinstance(spec, tuple) and len(spec) == 2 and not callable(spec[0]):
        # (m_left, m_right) of tuples
        ml, mr = spec
        chosen = ml if side == "left" else mr
        return lambda y: chosen
    if isinstance(spec, tuple) and len(spec) == 2 and callable(spec[0]):
        ml, mr = spec
        chosen = ml if side == "left" else mr
        return lambda y: chosen(y * a)
    raise ValueError(f"Unrecognized lead_magnetization spec: {spec!r}")


def _attach_leads(syst, lat, Nx, Ny, a, H_on, Hx, Hy, *,
                  magnetization, G_E, G_H, lead_axis, lead_magnetization):
    if lead_axis == "x":
        sym = kwant.TranslationalSymmetry((-a, 0))
        # Left lead asymptotes the magnetization at x=0; right at x=Nx-1.
        m_left = _resolve_lead_m(lead_magnetization, magnetization, 0, Ny, a, "left")
        m_right = _resolve_lead_m(lead_magnetization, magnetization, Nx - 1, Ny, a, "right")
        _build_x_lead(syst, lat, Ny, sym, H_on, Hx, Hy, m_left, G_E, G_H)
        sym_r = kwant.TranslationalSymmetry((a, 0))
        _build_x_lead(syst, lat, Ny, sym_r, H_on, Hx, Hy, m_right, G_E, G_H)
    else:
        sym = kwant.TranslationalSymmetry((0, -a))
        m_bot = _resolve_lead_m(lead_magnetization, magnetization, 0, Nx, a, "left")
        m_top = _resolve_lead_m(lead_magnetization, magnetization, Ny - 1, Nx, a, "right")
        _build_y_lead(syst, lat, Nx, sym, H_on, Hx, Hy, m_bot, G_E, G_H)
        sym_r = kwant.TranslationalSymmetry((0, a))
        _build_y_lead(syst, lat, Nx, sym_r, H_on, Hx, Hy, m_top, G_E, G_H)


def _build_x_lead(syst, lat, Ny, sym, H_on, Hx, Hy, m_func, G_E, G_H):
    lead = kwant.Builder(sym)
    for y in range(Ny):
        mat = H_on.copy()
        m = m_func(y)
        if m is not None and (G_E != 0.0 or G_H != 0.0):
            mat = mat + exchange_matrix(m, G_E=G_E, G_H=G_H)
        lead[lat(0, y)] = mat
    for y in range(Ny - 1):
        lead[lat(0, y + 1), lat(0, y)] = Hy
    for y in range(Ny):
        lead[lat(1, y), lat(0, y)] = Hx
    syst.attach_lead(lead)


def _build_y_lead(syst, lat, Nx, sym, H_on, Hx, Hy, m_func, G_E, G_H):
    lead = kwant.Builder(sym)
    for x in range(Nx):
        mat = H_on.copy()
        m = m_func(x)
        if m is not None and (G_E != 0.0 or G_H != 0.0):
            mat = mat + exchange_matrix(m, G_E=G_E, G_H=G_H)
        lead[lat(x, 0)] = mat
    for x in range(Nx - 1):
        lead[lat(x + 1, 0), lat(x, 0)] = Hx
    for x in range(Nx):
        lead[lat(x, 1), lat(x, 0)] = Hy
    syst.attach_lead(lead)


def orbital_to_site_grid(Nx, Ny):
    """Return an int array of shape (Nx, Ny) mapping (x,y) → first orbital index.

    Useful for building the orbital_grid that ``sc.poisson_solve_2d`` expects
    when combining BHZ with the SC-Poisson loop. The convention here matches
    the loop in ``make_bhz_system``: site (x, y) is the (x*Ny + y)-th site,
    and each site has 4 orbitals.
    """
    grid = np.zeros((Nx, Ny), dtype=int)
    for x in range(Nx):
        for y in range(Ny):
            grid[x, y] = (x * Ny + y) * 4
    return grid
