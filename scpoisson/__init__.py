"""Self-consistent Poisson solver for Kwant quantum transport systems.

Modules
-------
density   : NEGF charge density (equilibrium + non-equilibrium)
poisson   : 1D / 2D Poisson solvers with Dirichlet/Neumann BCs
mixing    : Mixing schemes for the self-consistent iteration
sc        : Top-level self-consistent loop orchestrator
bhz       : Built-in BHZ / Chern-insulator (QAH) model with magnetization
"""

from . import density, poisson, mixing, sc, bhz

__all__ = ["density", "poisson", "mixing", "sc", "bhz"]
__version__ = "0.0.2"
