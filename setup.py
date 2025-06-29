from setuptools import setup, find_packages

setup(
    name="med-vllm",
    version="0.1.0",
    packages=find_packages(include=['medvllm', 'medvllm.*']),
    install_requires=[
        'torch>=2.4.0',
        'triton>=3.0.0',
        'transformers>=4.51.0',
        'xxhash',
    ],
    extras_require={
        'all': ['flash-attn'],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
        ],
    },
)
