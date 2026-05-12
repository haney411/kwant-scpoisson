"""Smoke test: 1D tight-binding chain with leads, compute transmission(E).

Note: kwant 1.5.0 has a known incompatibility with newer scipy at energies
where the lead has zero propagating modes (out of band) — the mode-finding
code calls la.solve with a non-square matrix. We stick to in-band energies
here. See notes/kwant_lead_bug.md.
"""
import kwant
import numpy as np

lat = kwant.lattice.chain(norbs=1)
N = 20
t = 1.0

syst = kwant.Builder()
for i in range(N):
    syst[lat(i)] = 0.0
for i in range(N - 1):
    syst[lat(i), lat(i + 1)] = -t

sym = kwant.TranslationalSymmetry((-1,))
lead = kwant.Builder(sym)
lead[lat(0)] = 0.0
lead[lat(0), lat(1)] = -t
syst.attach_lead(lead)
syst.attach_lead(lead.reversed())

syst = syst.finalized()

# Stay strictly inside the band (-2t, 2t) = (-2, 2).
energies = np.linspace(-1.8, 1.8, 19)
T = np.array([kwant.smatrix(syst, E).transmission(1, 0) for E in energies])

print(f"T at sampled energies: min={T.min():.6f} max={T.max():.6f} mean={T.mean():.6f}")
assert np.allclose(T, 1.0, atol=1e-6), f"in-band transmission not unity: {T}"
print("OK: 1D chain transmission smoke test passed (in-band).")
