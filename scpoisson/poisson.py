"""1D Poisson solver with Dirichlet / Neumann boundary conditions.

Solves the linear Poisson equation in 1D on a uniform grid:

    -d/dx [eps(x) dV/dx] = rho(x) / eps0

Returns electrostatic potential V(x) at each grid point. For SC-Poisson
in kwant, the grid points coincide with the longitudinal sites of the
scattering region.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


EPS0_SI = 8.8541878128e-12  # F/m   (= C²/(N·m²))


def solve_1d(
    rho,
    dx,
    eps_r=1.0,
    bc_left=("dirichlet", 0.0),
    bc_right=("dirichlet", 0.0),
    eps0=EPS0_SI,
    charge_e=1.602176634e-19,
):
    """Solve 1D Poisson on a uniform grid.

    Parameters
    ----------
    rho : (N,) array
        Charge density at each grid point. **Units**: charge per unit
        length per unit area would be the full 3D answer; here for a 1D
        Poisson over a quasi-2D wire treated as effective 1D, the user
        controls units by their choice of eps0 and rho. For the toy
        problems in this codebase we typically work in natural units
        where the prefactor is absorbed (see `solve_1d_natural`).
    dx : float
        Grid spacing (real-space distance between sites).
    eps_r : float or (N,) array
        Relative permittivity (uniform or per-site).
    bc_left, bc_right : tuple ``(kind, value)``
        ``kind`` is "dirichlet" (V = value) or "neumann" (dV/dx = value).
    eps0 : float
        Vacuum permittivity (default SI).
    charge_e : float
        Elementary charge (default SI). Used to convert ``rho`` (number
        density) to charge density rho_q = -e * rho_electron + ... done
        by the caller.

    Returns
    -------
    V : (N,) array
        Electrostatic potential at each grid point, same units as
        rho * dx² / eps.
    """
    rho = np.asarray(rho, dtype=float)
    N = rho.size
    if np.isscalar(eps_r):
        eps_r = np.full(N, eps_r, dtype=float)
    else:
        eps_r = np.asarray(eps_r, dtype=float)
        if eps_r.size != N:
            raise ValueError("eps_r length must match rho length")

    # Build the FD Laplacian with non-uniform eps via flux-conservative form:
    #   ( eps_{i+1/2} (V_{i+1}-V_i) - eps_{i-1/2} (V_i-V_{i-1}) ) / dx² = -rho_i/eps0
    # i.e. A V = b, with b_i = rho_i * dx² / eps0.
    eps_half = 0.5 * (eps_r[:-1] + eps_r[1:])  # length N-1 (between sites)

    # Diagonals
    main = np.zeros(N)
    upper = np.zeros(N - 1)
    lower = np.zeros(N - 1)
    for i in range(N):
        if i == 0:
            main[i] += eps_half[0]
            upper[i] = -eps_half[0]
        elif i == N - 1:
            main[i] += eps_half[-1]
            lower[i - 1] = -eps_half[-1]
        else:
            main[i] += eps_half[i - 1] + eps_half[i]
            upper[i] = -eps_half[i]
            lower[i - 1] = -eps_half[i - 1]

    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="lil")
    b = rho * dx ** 2 / eps0

    # Apply BCs by modifying rows 0 and N-1.
    def apply_bc(row, side):
        kind, val = (bc_left if side == "left" else bc_right)
        i = 0 if side == "left" else N - 1
        if kind == "dirichlet":
            # Wipe the row and set V_i = val.
            A.rows[i] = [i]
            A.data[i] = [1.0]
            b[i] = val
        elif kind == "neumann":
            # dV/dx |_i = val.
            # One-sided FD: (V_{i±1} - V_i)/dx = ±val   (right-handed)
            if side == "left":
                # (V_1 - V_0)/dx = val
                A.rows[0] = [0, 1]
                A.data[0] = [-1.0 / dx, 1.0 / dx]
                b[0] = val
            else:
                # (V_{N-1} - V_{N-2})/dx = val
                A.rows[N - 1] = [N - 2, N - 1]
                A.data[N - 1] = [-1.0 / dx, 1.0 / dx]
                b[N - 1] = val
        else:
            raise ValueError(f"unknown BC kind: {kind}")

    apply_bc(0, "left")
    apply_bc(N - 1, "right")

    A = A.tocsc()
    V = spla.spsolve(A, b)
    return V


def solve_2d_natural(
    rho,
    dx,
    dy=None,
    eps_r=1.0,
    bc_x_left=("dirichlet", 0.0),
    bc_x_right=("dirichlet", 0.0),
    bc_y_bot=("neumann", 0.0),
    bc_y_top=("neumann", 0.0),
    coupling=1.0,
):
    """Solve -∇·(ε ∇V) = coupling · ρ on an Nx × Ny rectangular grid.

    Natural-units form (analog of `solve_1d_natural`): no physical
    constants, the `coupling` knob sets the Hartree strength.

    Parameters
    ----------
    rho : (Nx, Ny) array
        Source (electron-number-density units, e.g. ``n_e - n_background``).
    dx, dy : float
        Grid spacings in x and y. Default ``dy = dx``.
    eps_r : float or (Nx, Ny) array
        Relative permittivity, uniform or per-site.
    bc_x_left, bc_x_right : (kind, value)
        Boundary condition on the longitudinal (x) edges. Defaults to
        Dirichlet V=0 (anchoring the potential, as usual for transport
        regions sandwiched between metallic leads).
    bc_y_bot, bc_y_top : (kind, value)
        Boundary condition on the transverse (y) edges. Defaults to
        Neumann ∂V/∂y = 0 (free-standing quasi-2D, no normal field).
    coupling : float
        Hartree coupling strength.

    Returns
    -------
    V : (Nx, Ny) array
        Self-consistent potential energy felt by electrons.

    Conventions
    -----------
    Index ordering of the flattened system is row-major in x: orbital
    ``k = x*Ny + y``. Boundary condition ``"neumann"`` with value v means
    (V[1, y] − V[0, y])/dx = v at the left edge, and analogously for the
    other three; v=0 is the zero-flux condition.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError("rho must be 2D")
    Nx, Ny = rho.shape
    if dy is None:
        dy = dx
    if np.isscalar(eps_r):
        eps = np.full((Nx, Ny), float(eps_r))
    else:
        eps = np.asarray(eps_r, dtype=float)
        if eps.shape != rho.shape:
            raise ValueError("eps_r shape mismatch with rho")

    # Half-step permittivities between neighboring grid points.
    eps_x_half = 0.5 * (eps[:-1, :] + eps[1:, :])   # shape (Nx-1, Ny), at (x+1/2, y)
    eps_y_half = 0.5 * (eps[:, :-1] + eps[:, 1:])   # shape (Nx, Ny-1), at (x, y+1/2)

    N = Nx * Ny

    def k(x, y):
        return x * Ny + y

    A = sp.lil_matrix((N, N))
    b = (coupling * rho * dx * dy).flatten()
    # Note: the source has units of (flux / area)·dxdy = total flux per cell
    # for a finite-volume formulation.

    for x in range(Nx):
        for y in range(Ny):
            kc = k(x, y)
            # We accumulate the 5-point stencil with flux-conservative
            # coefficients, then overwrite the row for boundary edges if
            # the BC is Dirichlet (Neumann is enforced via a ghost-flux
            # contribution to the source instead).
            diag = 0.0

            # East face (toward +x)
            if x < Nx - 1:
                c = eps_x_half[x, y] * dy / dx
                A[kc, k(x + 1, y)] -= c
                diag += c
            else:
                # Right (x=Nx-1) boundary
                kind, val = bc_x_right
                if kind == "neumann":
                    # Outflux = eps·(∂V/∂x_outward)·dy = eps·val·dy
                    # Outward x-normal at x=Nx-1 is +x.
                    eps_face = eps[x, y]
                    b[kc] -= eps_face * val * dy
                # Dirichlet handled below by row replacement.

            # West face (toward -x)
            if x > 0:
                c = eps_x_half[x - 1, y] * dy / dx
                A[kc, k(x - 1, y)] -= c
                diag += c
            else:
                kind, val = bc_x_left
                if kind == "neumann":
                    # Outward normal at x=0 is -x, so ∂V/∂x_out = -∂V/∂x
                    # = val means ∂V/∂x = -val. The outward flux through
                    # the face is eps·(∂V/∂x_out)·dy = eps·val·dy.
                    eps_face = eps[x, y]
                    b[kc] -= eps_face * val * dy

            # North face (+y)
            if y < Ny - 1:
                c = eps_y_half[x, y] * dx / dy
                A[kc, k(x, y + 1)] -= c
                diag += c
            else:
                kind, val = bc_y_top
                if kind == "neumann":
                    eps_face = eps[x, y]
                    b[kc] -= eps_face * val * dx

            # South face (-y)
            if y > 0:
                c = eps_y_half[x, y - 1] * dx / dy
                A[kc, k(x, y - 1)] -= c
                diag += c
            else:
                kind, val = bc_y_bot
                if kind == "neumann":
                    eps_face = eps[x, y]
                    b[kc] -= eps_face * val * dx

            A[kc, kc] = diag

    # Now overwrite rows for Dirichlet boundary points.
    def set_dirichlet(x, y, val):
        kc = k(x, y)
        # Zero the row
        A.rows[kc] = [kc]
        A.data[kc] = [1.0]
        b[kc] = val

    # x edges
    kind, val = bc_x_left
    if kind == "dirichlet":
        for y in range(Ny):
            set_dirichlet(0, y, val)
    elif kind != "neumann":
        raise ValueError(f"unknown BC kind on x-left: {kind}")
    kind, val = bc_x_right
    if kind == "dirichlet":
        for y in range(Ny):
            set_dirichlet(Nx - 1, y, val)
    elif kind != "neumann":
        raise ValueError(f"unknown BC kind on x-right: {kind}")
    # y edges
    kind, val = bc_y_bot
    if kind == "dirichlet":
        for x in range(Nx):
            set_dirichlet(x, 0, val)
    elif kind != "neumann":
        raise ValueError(f"unknown BC kind on y-bot: {kind}")
    kind, val = bc_y_top
    if kind == "dirichlet":
        for x in range(Nx):
            set_dirichlet(x, Ny - 1, val)
    elif kind != "neumann":
        raise ValueError(f"unknown BC kind on y-top: {kind}")

    # Check for the all-Neumann singular case: the Laplacian is then only
    # defined up to an additive constant, and the source must integrate to
    # the boundary-flux total. We pin V[0,0]=0 to remove the null space.
    if (bc_x_left[0] == bc_x_right[0] == bc_y_bot[0] == bc_y_top[0] == "neumann"):
        set_dirichlet(0, 0, 0.0)

    A = A.tocsc()
    V = spla.spsolve(A, b)
    return V.reshape(Nx, Ny)


def solve_1d_natural(rho_e, dx, eps_r=1.0,
                     bc_left=("dirichlet", 0.0),
                     bc_right=("dirichlet", 0.0),
                     coupling=1.0):
    """Natural-units 1D Poisson for toy tight-binding tests.

    Treats ``rho_e`` as a dimensionless electron density and returns a
    "potential energy" V (eV-like) that can be added directly to the
    Hamiltonian onsite. The single ``coupling`` parameter sets the
    Hartree strength (e²/(4πε₀ε_r·dx) in real units, but here just a
    knob).

    -d²V/dx² = coupling · rho_e
    """
    rho_e = np.asarray(rho_e, dtype=float)
    N = rho_e.size
    if np.isscalar(eps_r):
        eps_r = np.full(N, eps_r, dtype=float)
    eps_half = 0.5 * (eps_r[:-1] + eps_r[1:])

    main = np.zeros(N)
    upper = np.zeros(N - 1)
    lower = np.zeros(N - 1)
    for i in range(N):
        if i == 0:
            main[i] += eps_half[0]
            upper[i] = -eps_half[0]
        elif i == N - 1:
            main[i] += eps_half[-1]
            lower[i - 1] = -eps_half[-1]
        else:
            main[i] += eps_half[i - 1] + eps_half[i]
            upper[i] = -eps_half[i]
            lower[i - 1] = -eps_half[i - 1]

    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="lil")
    b = coupling * rho_e * dx ** 2

    def apply_bc(side):
        kind, val = (bc_left if side == "left" else bc_right)
        i = 0 if side == "left" else N - 1
        if kind == "dirichlet":
            A.rows[i] = [i]
            A.data[i] = [1.0]
            b[i] = val
        elif kind == "neumann":
            if side == "left":
                A.rows[0] = [0, 1]
                A.data[0] = [-1.0 / dx, 1.0 / dx]
                b[0] = val
            else:
                A.rows[N - 1] = [N - 2, N - 1]
                A.data[N - 1] = [-1.0 / dx, 1.0 / dx]
                b[N - 1] = val
        else:
            raise ValueError(f"unknown BC kind: {kind}")

    apply_bc("left")
    apply_bc("right")
    A = A.tocsc()
    V = spla.spsolve(A, b)
    return V
