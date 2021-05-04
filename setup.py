from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pytorch-bnns",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    version="0.0.1",
    description=(
        "PyTorch implementation of BNNs"
    ),
    install_requires=[
        "torch",
        "torchvision",
        "kornia==0.4.1",
        "torch-lucent==0.1.7",
        "tqdm",
        "numpy",
        "ipython",
        "pillow",
        "future",
        "decorator",
        "pytest",
        "pytest-mock",
        "coverage",
        "coveralls",
        "scikit-learn"
    ],
)
