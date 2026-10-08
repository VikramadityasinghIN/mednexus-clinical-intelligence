"""Offline test runner (no pytest needed).

Usage:
    python run_tests.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tests.test_all as T

TESTS = [getattr(T, n) for n in sorted(dir(T)) if n.startswith("test_")]


def main():
    passed, failed = 0, []
    t0 = time.time()
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
            failed.append(fn.__name__)
    dt = time.time() - t0
    print(f"\n{passed}/{passed + len(failed)} tests passed in {dt:.1f}s")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
