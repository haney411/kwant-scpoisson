# Kwant 1.5.0 lead-mode finder bug (compat with scipy ≥1.14)

**Symptom:** at energies outside the lead band (only evanescent eigenvalues), `kwant.smatrix` / `kwant.greens_function` raise:

```
ValueError: Expected square matrix, got a1.shape=(2, 1)
```

from `kwant/physics/leads.py:687`:

```python
psi[:, indx] = la.solve(r.T, psi[:, indx].T).T
```

**Root cause:** `la.qr(full_psi[:, indx], mode='economic')` returns a non-square `r` when the lead has very few orbitals and all transfer-matrix eigenvalues are flagged as belonging to the same degenerate cluster (e.g. single-orbital 1D chain at out-of-band energies — all evanescent λ have phase 0 or π, so the "all eigenvalues equal" branch on leads.py:651 fires). Newer scipy versions enforce `la.solve` square matrix; older scipy silently called lstsq.

**Practical workaround for the SC-Poisson project:**

1. **Avoid exact band edges in energy meshes.** For real-axis integration of LDOS, sample energies inside the band; the band-edge contribution is measure-zero in the integral.
2. **Prefer contour integration** for the equilibrium part of the density (semicircle in upper half-plane from low energy below the band up to E_F). Complex energies away from the real axis don't hit the bug. This is the standard approach (Lake, Datta, Brandbyge, ...) and converges faster anyway.
3. **For real-axis evaluation in the bias window** (non-eq part), stay strictly inside the band of each lead.

**Permanent fix candidates:**

- Patch kwant locally: replace line 687 with
  ```python
  if r.shape[0] == r.shape[1]:
      psi[:, indx] = la.solve(r.T, psi[:, indx].T).T
  else:
      psi[:, indx] = la.lstsq(r.T, psi[:, indx].T)[0].T
  ```
  But out-of-band physics there is degenerate anyway, so the workaround may not give meaningful "modes" — verify with kwant maintainers.
- Upstream issue / kwant gitlab.kwant-project.org; check whether 1.5.x has a fix.

For now, document and proceed; we will revisit only if we need values exactly at band edges.
