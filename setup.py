from setuptools import setup, find_packages

setup(
    name="med-vllm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
        "transformers>=4.51.0",
        "xxhash",
    ],
    extras_require={
        'all': [
            'flash-attn',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=0.990',
            'flake8>=5.0.0',
        ],
    },
    python_requires='>=3.10,<3.13',
)
