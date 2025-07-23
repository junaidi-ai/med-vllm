from setuptools import find_packages, setup

# Read requirements from requirements-medical.txt
with open("requirements-medical.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="med-vllm",
    version="0.1.0",
    packages=find_packages(include=["medvllm", "medvllm.*"]),
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
        "transformers>=4.51.0",
        "xxhash",
        "safetensors>=0.3.1",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    ],
    extras_require={
        "all": ["flash-attn"],
        "medical": requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    package_data={
        "medvllm": ["configs/models/*.yaml"],
    },
    include_package_data=True,
)
