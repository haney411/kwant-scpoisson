"""Tests for the BHZ + magnetization model."""

import numpy as np

from scpoisson import bhz


# ---------- low-level matrix tests ----------

def _bloch_h(kx, ky, *, A, B, C, D, M, a=1.0):
    """Reference Bloch Hamiltonian assembled from the building blocks."""
    H_on = bhz.bhz_onsite_matrix(A, B, C, D, M, a=a)
    Hx = bhz.bhz_hopx_matrix(A, B, D, a=a)
    Hy = bhz.bhz_hopy_matrix(A, B, D, a=a)
    return (
        H_on
        + Hx * np.exp(1j * kx * a) + Hx.conj().T * np.exp(-1j * kx * a)
        + Hy * np.exp(1j * ky * a) + Hy.conj().T * np.exp(-1j * ky * a)
    )


def test_bhz_matrices_hermitian():
    H = _bloch_h(0.3, -0.2, A=1.0, B=1.0, C=0.1, D=0.05, M=0.7)
    assert np.allclose(H, H.conj().T, atol=1e-14)


def test_bhz_gamma_point_matches_continuum():
    A, B, C, D, M = 1.2, 0.8, 0.3, 0.1, -0.6
    H0 = _bloch_h(0.0, 0.0, A=A, B=B, C=C, D=D, M=M)
    beta = np.diag([1.0, -1.0, 1.0, -1.0])
    expected = C * np.eye(4) + M * beta
    assert np.allclose(H0, expected, atol=1e-13)


def test_bhz_small_k_recovers_continuum():
    """Expand H(k) at small k and check the lattice → continuum match."""
    A, B, C, D, M = 1.0, 1.0, 0.0, 0.0, 0.5
    beta = np.diag([1.0, -1.0, 1.0, -1.0])
    alpha_x = np.array(
        [[0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, -1],
         [0, 0, -1, 0]], dtype=complex,
    )
    alpha_y = np.array(
        [[0, -1j, 0, 0],
         [1j, 0, 0, 0],
         [0, 0, 0, -1j],
         [0, 0, 1j, 0]], dtype=complex,
    )

    for kx, ky in [(1e-3, 0.0), (0.0, 1e-3), (5e-4, 5e-4)]:
        H_lat = _bloch_h(kx, ky, A=A, B=B, C=C, D=D, M=M)
        H_cont = (
            (C - D * (kx**2 + ky**2)) * np.eye(4)
            + (M - B * (kx**2 + ky**2)) * beta
            + A * kx * alpha_x + A * ky * alpha_y
        )
        # 4th-order corrections are tiny at these k.
        assert np.allclose(H_lat, H_cont, atol=1e-9), (kx, ky)


def test_gap_at_gamma_equals_2M():
    """With no exchange and C=0, the gap at Γ is 2|M|."""
    for M in (0.3, 0.7, 1.2):
        H0 = _bloch_h(0.0, 0.0, A=1.0, B=1.0, C=0.0, D=0.0, M=M)
        evals = np.sort(np.linalg.eigvalsh(H0))
        # Expect two at -M and two at +M.
        assert np.isclose(evals[0], -M)
        assert np.isclose(evals[1], -M)
        assert np.isclose(evals[2], +M)
        assert np.isclose(evals[3], +M)


def test_exchange_splits_blocks_correctly():
    """Uniform m_z with G_E ≠ G_H shifts the four Γ-point energies by ±m_z·G_E/H.

    Comparison is against the Bloch H at k=0 (where the BHZ kinetic terms
    collapse to C·I + M·β) plus the exchange. The bare on-site matrix has
    extra −4D/a², −4B/a² shifts that get cancelled by the +x, +y hoppings.
    """
    A, B, M, G_E, G_H, m0 = 1.0, 1.0, 0.5, 2.0, 0.0, 0.7
    H = _bloch_h(0.0, 0.0, A=A, B=B, C=0.0, D=0.0, M=M)
    H = H + bhz.exchange_matrix((0.0, 0.0, m0), G_E=G_E, G_H=G_H)
    expected_diag = np.array([
        M + m0 * G_E,
        -M + m0 * G_H,
        M - m0 * G_E,
        -M - m0 * G_H,
    ])
    assert np.allclose(np.diag(H).real, expected_diag, atol=1e-13)
    off = H - np.diag(np.diag(H))
    assert np.allclose(off, 0.0, atol=1e-13)


def test_exchange_in_plane_couples_spins():
    """m_x ≠ 0 should produce off-diagonal elements between ↑ and ↓ sectors."""
    H_exch = bhz.exchange_matrix((1.0, 0.0, 0.0), G_E=1.5, G_H=0.5)
    # E↑ ↔ E↓ element should be G_E; H↑ ↔ H↓ should be G_H.
    assert np.isclose(H_exch[0, 2], 1.5)
    assert np.isclose(H_exch[1, 3], 0.5)
    assert np.isclose(H_exch[2, 0], 1.5)
    assert np.isclose(H_exch[3, 1], 0.5)
    # Diagonal stays zero.
    assert np.allclose(np.diag(H_exch), 0.0, atol=1e-14)
    # Hermitian.
    assert np.allclose(H_exch, H_exch.conj().T, atol=1e-14)


# ---------- domain wall helpers ----------

def test_domain_wall_ising_profile():
    m_func = bhz.domain_wall_magnetization(
        axis="x", center=5.0, width=2.0, m0=1.0, kind="ising",
    )
    # At center, m_z = 0; far below, m_z → -1; far above, m_z → +1.
    assert m_func(5.0, 0.0) == (0.0, 0.0, 0.0)
    assert np.isclose(m_func(-50.0, 0.0)[2], -1.0)
    assert np.isclose(m_func(+50.0, 0.0)[2], +1.0)
    # No in-plane component for the ising wall.
    assert np.isclose(m_func(6.5, 3.2)[0], 0.0)
    assert np.isclose(m_func(6.5, 3.2)[1], 0.0)


def test_domain_wall_bloch_neel_have_in_plane_component():
    """Bloch / Néel walls have an in-plane sech profile peaked at the center."""
    for kind, idx in (("bloch", 1), ("neel", 0)):
        m_func = bhz.domain_wall_magnetization(
            axis="x", center=0.0, width=1.0, m0=1.0, kind=kind,
        )
        peak = m_func(0.0, 0.0)[idx]
        tail = m_func(10.0, 0.0)[idx]
        assert np.isclose(peak, 1.0)
        assert abs(tail) < 1e-3
        # The other in-plane component stays zero.
        other = 0 if idx == 1 else 1
        assert np.isclose(m_func(0.0, 0.0)[other], 0.0)


# ---------- kwant system builder ----------

def test_make_bhz_system_no_leads_hermitian():
    syst = bhz.make_bhz_system(Nx=6, Ny=5, M=0.5, leads=False)
    H = syst.hamiltonian_submatrix().toarray() if hasattr(
        syst.hamiltonian_submatrix(), "toarray"
    ) else syst.hamiltonian_submatrix()
    assert H.shape == (6 * 5 * 4, 6 * 5 * 4)
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_make_bhz_system_with_domain_wall_builds():
    m_func = bhz.domain_wall_magnetization(
        axis="x", center=4.0, width=1.5, m0=1.0, kind="ising",
    )
    syst = bhz.make_bhz_system(
        Nx=8, Ny=4, M=0.5, G_E=2.0, G_H=-0.3,
        magnetization=m_func, leads=True, lead_axis="x",
    )
    # Two leads attached.
    assert len(syst.leads) == 2
    # System Hamiltonian Hermitian.
    H = syst.hamiltonian_submatrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_v_onsite_shifts_spectrum():
    """Adding V_onsite shifts the full spectrum by V uniformly per site."""
    Nx, Ny = 4, 3
    syst0 = bhz.make_bhz_system(Nx=Nx, Ny=Ny, M=0.5, leads=False)
    V = 0.4 * np.ones(Nx * Ny)
    syst1 = bhz.make_bhz_system(
        Nx=Nx, Ny=Ny, M=0.5, V_onsite=V, leads=False,
    )
    H0 = syst0.hamiltonian_submatrix()
    H1 = syst1.hamiltonian_submatrix()
    # H1 = H0 + 0.4 * I
    assert np.allclose(H1 - H0, 0.4 * np.eye(H0.shape[0]), atol=1e-12)


def test_qah_block_inversion():
    """At m_z = ±M/G_E one spin block's Γ-point mass closes — the QAH transition.

    With G_E ≠ 0, G_H = 0, the four Γ energies are M ± m_z G_E (upper block)
    and ±(-M) (lower block, untouched). One eigenvalue passes through 0 at the
    critical m_z while the others stay finite — the spin-resolved gap closure
    that defines the Chern-changing transition.
    """
    M, G_E, G_H = 0.5, 2.0, 0.0
    m_crit = -M / G_E                       # closes E↑/H↑ block (eigenvalue at 0)
    H = _bloch_h(0.0, 0.0, A=1, B=1, C=0, D=0, M=M)
    H_crit = H + bhz.exchange_matrix((0, 0, m_crit), G_E=G_E, G_H=G_H)
    evals_crit = np.sort(np.linalg.eigvalsh(H_crit))
    # Expected: [-M, -M, 0, +2M] when m_z = -M/G_E.
    expected = np.array([-M, -M, 0.0, 2 * M])
    assert np.allclose(evals_crit, expected, atol=1e-12)

    # Away from m_crit, the gap reopens — no eigenvalue should be near zero.
    H_off = H + bhz.exchange_matrix((0, 0, m_crit + 0.2), G_E=G_E, G_H=G_H)
    evals_off = np.linalg.eigvalsh(H_off)
    assert np.min(np.abs(evals_off)) > 0.1
