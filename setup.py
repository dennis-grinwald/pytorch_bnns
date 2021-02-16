from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bayesian-actmax",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    version="0.0.1",
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
