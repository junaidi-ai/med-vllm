#!/usr/bin/env python3
"""
Environment and package verification script.
"""

import sys
import os
import logging
import importlib
import pkg_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def check_package_installed(package_name: str) -> bool:
    """Check if a package is installed and return its version."""
    try:
        version = pkg_resources.get_distribution(package_name).version
        logger.info(f"✓ {package_name} (v{version}) is installed")
        return True
    except pkg_resources.DistributionNotFound:
        logger.error(f"✗ {package_name} is NOT installed")
        return False
    except Exception as e:
        logger.error(f"Error checking {package_name}: {e}")
        return False


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        logger.info(f"✓ Successfully imported {module_name}")
        if package_name:
            version = pkg_resources.get_distribution(package_name).version
            logger.info(f"  {package_name} version: {version}")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        return False


def check_medvllm_structure() -> None:
    """Check the medvllm package structure."""
    logger.info("\nChecking medvllm package structure...")
    medvllm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "medvllm"))

    if not os.path.exists(medvllm_path):
        logger.error(f"✗ medvllm package not found at {medvllm_path}")
        return

    logger.info(f"✓ medvllm package found at {medvllm_path}")

    # Check important files and directories
    required_files = [
        "__init__.py",
        "models/__init__.py",
        "models/adapters/__init__.py",
        "models/adapters/biobert.py",
        "models/adapters/clinicalbert.py",
        "models/adapters/base.py",
    ]

    for rel_path in required_files:
        full_path = os.path.join(medvllm_path, rel_path)
        if os.path.exists(full_path):
            logger.info(f"✓ Found: {rel_path}")
        else:
            logger.error(f"✗ Missing: {rel_path}")


def main() -> int:
    """Main function to run environment checks."""
    logger.info("=" * 80)
    logger.info("Environment and Package Verification")
    logger.info("=" * 80)

    # Check Python version
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")

    # Check required packages
    logger.info("\nChecking required packages...")
    required_packages = ["torch", "transformers", "numpy", "tqdm", "scikit-learn"]

    all_installed = True
    for package in required_packages:
        if not check_package_installed(package):
            all_installed = False

    # Check imports
    logger.info("\nChecking imports...")
    check_import("torch", "torch")
    check_import("transformers", "transformers")
    check_import("numpy", "numpy")
    check_import("sklearn", "scikit-learn")

    # Check medvllm imports
    logger.info("\nChecking medvllm imports...")
    try:
        import medvllm

        logger.info("✓ Successfully imported medvllm")
        logger.info(f"medvllm path: {os.path.dirname(medvllm.__file__)}")
    except ImportError as e:
        logger.error(f"✗ Failed to import medvllm: {e}")

    # Check medvllm structure
    check_medvllm_structure()

    logger.info("\nEnvironment check completed.")
    return 0 if all_installed else 1


if __name__ == "__main__":
    sys.exit(main())
