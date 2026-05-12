"""Run all tests in tests/ and report a pass/fail summary."""

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples"))

test_modules = [
    "test_density",
    "test_poisson",
    "test_sc_loop",
    "test_nonequilibrium",
    "test_ribbon",
    "test_ribbon_2d",
]

# Make tests/ importable
sys.path.insert(0, HERE)

n_pass = n_fail = 0
failures = []
for modname in test_modules:
    mod = importlib.import_module(modname)
    for name in dir(mod):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        full = f"{modname}.{name}"
        try:
            fn()
            print(f"  PASS  {full}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {full}: {e}")
            failures.append((full, traceback.format_exc()))
            n_fail += 1

print()
print(f"{n_pass} passed, {n_fail} failed")
for full, tb in failures:
    print()
    print(f"--- {full} ---")
    print(tb)
sys.exit(0 if n_fail == 0 else 1)
