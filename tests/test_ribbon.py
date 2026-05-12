"""Regression test for the quasi-2D ribbon SC example.

Imports from examples/example_sc_ribbon to avoid duplicating the factory.
"""

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples"))

import numpy as np
import example_sc_ribbon as ribbon


def test_ribbon_neutral_zero_perturbation():
    """With no extra doping, V should remain ≈ 0."""
    V, n_col, n_ref = ribbon.run_ribbon_sc(
        N=10, W=3, mu=2.0, delta_bg_strength=0.0, max_iter=10,
    )
    assert np.max(np.abs(V)) < 1e-8, f"V should be ~0; got max|V| = {np.max(np.abs(V)):.3e}"


def test_ribbon_screening_response_sign():
    """Positive bg on right half → negative V on right → extra electrons."""
    V, n_col, n_ref = ribbon.run_ribbon_sc(
        N=12, W=3, mu=2.0, delta_bg_strength=0.04, max_iter=20,
    )
    # V should be more negative in the doped (right) half
    mid = len(V) // 2
    assert V[mid + 2 : -2].mean() < V[2 : mid - 1].mean(), \
        "V should be more negative on right (doped) side"
    # Electron density elevated on the right
    dn = n_col - n_ref
    assert dn[mid + 2 : -2].mean() > dn[2 : mid - 1].mean(), \
        "electron density should rise on the right (screening)"


if __name__ == "__main__":
    print("test_ribbon_neutral_zero_perturbation ...")
    test_ribbon_neutral_zero_perturbation()
    print("PASS\n")
    print("test_ribbon_screening_response_sign ...")
    test_ribbon_screening_response_sign()
    print("PASS\n")
    print("All ribbon tests passed.")
