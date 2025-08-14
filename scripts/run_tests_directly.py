import sys
import os
import unittest
import importlib.util

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_tests_from_module(module_path):
    """Load tests from a module given its path."""
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def main():
    # Load the test module
    test_file = "tests/medical/test_model_performance.py"
    test_suite = load_tests_from_module(test_file)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return non-zero exit code if any tests failed
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
