"""Run unittest suites and function-style registry contracts on CPU."""
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    suite = unittest.TestSuite()
    for folder in (ROOT / "tests", ROOT / "code/tests"):
        suite.addTests(unittest.TestLoader().discover(str(folder)))
    registry = runpy.run_path(str(ROOT / "code/tests/test_variant_registry.py"))
    for name, function in sorted(registry.items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun >= 99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
