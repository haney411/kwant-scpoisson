"""Self-consistent Poisson solver for Kwant quantum transport systems.

Modules
-------
density   : NEGF charge density (equilibrium + non-equilibrium)
poisson   : 1D Poisson solver with Dirichlet/Neumann BCs
mixing    : Mixing schemes for the self-consistent iteration
sc        : Top-level self-consistent loop orchestrator
"""

from . import density, poisson, mixing, sc

__all__ = ["density", "poisson", "mixing", "sc"]
__version__ = "0.0.1"
