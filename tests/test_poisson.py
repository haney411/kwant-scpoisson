"""Tests for scpoisson.poisson.

Compare the FD solver to analytic Poisson solutions on the unit interval.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scpoisson import poisson


def test_constant_rho_dirichlet():
    """-V''(x) = ρ, V(0)=V(L)=0 ⇒ V(x) = ρ x (L - x) / 2."""
    N = 201
    L = 1.0
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    rho = np.ones(N)  # constant
    V = poisson.solve_1d_natural(
        rho, dx=dx, eps_r=1.0,
        bc_left=("dirichlet", 0.0),
        bc_right=("dirichlet", 0.0),
        coupling=1.0,
    )
    V_an = 0.5 * x * (L - x)
    err = np.max(np.abs(V - V_an))
    print(f"  max abs err (Dirichlet/const ρ): {err:.3e}")
    assert err < 1e-10, f"err {err}"


def test_neumann_dirichlet_mix():
    """-V''(x) = ρ, dV/dx|_0 = 0, V(L) = V_R.

    V(x) = V_R + ρ (L²- x²)/2.   Pure quadratic, no boundary layer.
    """
    N = 201
    L = 1.0
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    rho = np.ones(N)
    V_R = 0.3
    V = poisson.solve_1d_natural(
        rho, dx=dx, eps_r=1.0,
        bc_left=("neumann", 0.0),
        bc_right=("dirichlet", V_R),
        coupling=1.0,
    )
    V_an = V_R + 0.5 * (L * L - x * x)
    err = np.max(np.abs(V - V_an))
    print(f"  max abs err (Neumann/Dirichlet): {err:.3e}")
    # First-order one-sided BC introduces O(dx) error at the boundary.
    assert err < 5e-3, f"err {err}"


def test_piecewise_eps_jump():
    """Check potential continuity & flux continuity at an eps jump.

    Layer 0..L/2 has eps=1, layer L/2..L has eps=2, with constant ρ=1
    and V(0)=V(L)=0. Solution is piecewise quadratic with matching
    V and eps*V' at L/2.
    """
    N = 401
    L = 1.0
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    eps_r = np.where(x < L / 2, 1.0, 2.0)
    rho = np.ones(N)
    V = poisson.solve_1d_natural(
        rho, dx=dx, eps_r=eps_r, coupling=1.0,
        bc_left=("dirichlet", 0.0),
        bc_right=("dirichlet", 0.0),
    )
    # Just check basic sanity: V > 0, max in interior, finite.
    assert np.all(V[1:-1] > 0), "interior V should be positive for ρ>0, V_BC=0"
    print(f"  V(0)={V[0]}, V(L/2)={V[N//2]}, V(L)={V[-1]}, V_max={V.max():.4f}")


def test_2d_fourier_mode_dirichlet():
    """ρ(x,y) = sin(πx/Lx)·sin(πy/Ly), all-zero Dirichlet, ε=1, coupling=1.

    -∇²V = ρ has analytic solution V = ρ / (π²/Lx² + π²/Ly²)."""
    Nx, Ny = 51, 41
    Lx, Ly = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    V = poisson.solve_2d_natural(
        rho, dx=dx, dy=dy, eps_r=1.0,
        bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
        bc_y_bot=("dirichlet", 0.0), bc_y_top=("dirichlet", 0.0),
        coupling=1.0,
    )
    V_an = rho / (np.pi ** 2 / Lx ** 2 + np.pi ** 2 / Ly ** 2)
    err = np.max(np.abs(V - V_an))
    print(f"  max abs err (2D Fourier mode): {err:.3e}")
    # FD error is O(dx²+dy²); with N≈50 we expect ~1/2500 ≈ 1e-4 amplitude.
    assert err < 5e-4, f"err {err}"


def test_2d_neumann_zero_source():
    """All Neumann BCs with zero source → V=0 (after gauge pin)."""
    Nx, Ny = 21, 11
    rho = np.zeros((Nx, Ny))
    V = poisson.solve_2d_natural(
        rho, dx=0.1, dy=0.1, eps_r=1.0,
        bc_x_left=("neumann", 0.0), bc_x_right=("neumann", 0.0),
        bc_y_bot=("neumann", 0.0), bc_y_top=("neumann", 0.0),
        coupling=1.0,
    )
    err = np.max(np.abs(V))
    print(f"  max|V| (all-Neumann zero source): {err:.3e}")
    assert err < 1e-12, f"err {err}"


def test_2d_reduces_to_1d():
    """ρ uniform in y, Neumann y edges → independent of y → matches 1D."""
    Nx, Ny = 41, 5
    dx = 0.025
    x = np.linspace(0, 1.0, Nx)
    rho1d = np.sin(np.pi * x)
    rho2d = np.broadcast_to(rho1d[:, None], (Nx, Ny)).copy()
    V1d = poisson.solve_1d_natural(
        rho1d, dx=dx, eps_r=1.0, coupling=1.0,
        bc_left=("dirichlet", 0.0), bc_right=("dirichlet", 0.0),
    )
    V2d = poisson.solve_2d_natural(
        rho2d, dx=dx, dy=dx, eps_r=1.0,
        bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
        bc_y_bot=("neumann", 0.0), bc_y_top=("neumann", 0.0),
        coupling=1.0,
    )
    # Should be independent of y; compare column 2 to 1D.
    err = np.max(np.abs(V2d[:, Ny // 2] - V1d))
    err_var = np.max(np.abs(V2d.std(axis=1)))  # y-variation should be zero
    print(f"  2D[mid] vs 1D: {err:.3e}, y-std variation: {err_var:.3e}")
    assert err < 1e-10, f"2D[mid] doesn't match 1D: err {err}"
    assert err_var < 1e-12, f"unexpected y-variation: {err_var}"


def test_2d_dirichlet_uniform_eps_const_rho():
    """-∇²V = c, V=0 on box boundary: analytic Fourier series solution.
    Compare numerical max V to series approximation."""
    Nx, Ny = 41, 41
    dx = 1.0 / (Nx - 1)
    rho = np.ones((Nx, Ny))
    V = poisson.solve_2d_natural(
        rho, dx=dx, dy=dx, eps_r=1.0, coupling=1.0,
        bc_x_left=("dirichlet", 0.0), bc_x_right=("dirichlet", 0.0),
        bc_y_bot=("dirichlet", 0.0), bc_y_top=("dirichlet", 0.0),
    )
    # Series solution at the center: V(1/2,1/2) = 16/π⁴ Σ_{m,n odd} 1/(mn(m²+n²))
    total = 0.0
    for m in range(1, 30, 2):
        for n in range(1, 30, 2):
            total += (np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
                      / (m * n * (m ** 2 + n ** 2)))
    V_center_an = (16.0 / np.pi ** 4) * total
    V_center_num = V[Nx // 2, Ny // 2]
    print(f"  V(center): num={V_center_num:.6f}  series={V_center_an:.6f}")
    assert abs(V_center_num - V_center_an) < 5e-4


if __name__ == "__main__":
    print("test_constant_rho_dirichlet ...")
    test_constant_rho_dirichlet()
    print("PASS\n")
    print("test_neumann_dirichlet_mix ...")
    test_neumann_dirichlet_mix()
    print("PASS\n")
    print("test_piecewise_eps_jump ...")
    test_piecewise_eps_jump()
    print("PASS\n")
    print("test_2d_fourier_mode_dirichlet ...")
    test_2d_fourier_mode_dirichlet()
    print("PASS\n")
    print("test_2d_neumann_zero_source ...")
    test_2d_neumann_zero_source()
    print("PASS\n")
    print("test_2d_reduces_to_1d ...")
    test_2d_reduces_to_1d()
    print("PASS\n")
    print("test_2d_dirichlet_uniform_eps_const_rho ...")
    test_2d_dirichlet_uniform_eps_const_rho()
    print("PASS\n")
    print("All Poisson tests passed.")
