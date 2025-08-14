import sys
import os
import unittest
import importlib.util
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_tests_from_module(module_path):
    """Load tests from a module given its path."""
    logger.info(f"Loading tests from {module_path}")
    try:
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        logger.debug(f"Module name: {module_name}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logger.error(f"Could not load spec for {module_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        logger.debug(f"Created module: {module}")

        # Try to import transformers first to check for issues
        try:
            import transformers

            logger.debug(f"Successfully imported transformers from: {transformers.__file__}")
            logger.debug("Successfully imported PreTrainedTokenizerBase")
        except Exception as e:
            logger.error(f"Failed to import transformers: {e}", exc_info=True)
            raise

        # Now load the test module
        spec.loader.exec_module(module)
        logger.debug(f"Successfully loaded module: {module}")

        return unittest.defaultTestLoader.loadTestsFromModule(module)
    except Exception as e:
        logger.error(f"Error loading tests from {module_path}: {e}", exc_info=True)
        raise


def main():
    logger.info("Starting test runner")

    # Load the test module
    test_file = os.path.abspath("tests/medical/test_model_performance.py")
    logger.info(f"Loading test file: {test_file}")

    test_suite = load_tests_from_module(test_file)
    if test_suite is None:
        logger.error("Failed to load test suite")
        sys.exit(1)

    logger.info(f"Running {test_suite.countTestCases()} test cases")

    # Run the tests with more detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Print a summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success: {result.wasSuccessful()}")

    # Return non-zero exit code if any tests failed
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
