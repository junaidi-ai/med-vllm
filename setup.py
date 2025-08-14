from setuptools import setup, find_packages

setup(
    name="med-vllm",
    version="0.1.0",
    packages=find_packages(include=["medvllm*"]),
    python_requires=">=3.10,<3.13",
    install_requires=["xxhash>=3.4.1", "pydantic>=2.0.0", "typing-extensions>=4.0.0"],
    extras_require={
        "all": [
            "torch>=2.4.0",
            "triton>=3.0.0",
            "transformers>=4.51.0",
            "flash-attn>=2.5.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "dev": [
            "black>=24.3.0",
            "isort>=5.13.2",
            "pylint>=3.1.0",
            "mypy>=1.11.1",
            "bandit>=1.7.7",
            "safety>=2.3.5",
            "pre-commit>=3.0.0",
            "types-requests",
            "types-pyyaml",
            "types-python-dateutil",
        ],
    },
)
