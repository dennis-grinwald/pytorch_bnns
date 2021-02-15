import lucent
from setuptools import setup, find_packages

version = lucent.__version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bayesian_activation_maximisation",
    packages=find_packages(exclude=[]),
    version=version,
    description=(
        "Uncertainty in activation maximisation"
    ),
    install_requires=[
        "torch>=1.5.0",
        "torchvision",
        "kornia==0.4.0",
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
