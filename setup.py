from setuptools import setup, find_packages
import os

# Read the README file for a long description.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LensingSSC",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for studying Super-Sample Covariance (SSC) effects in weak gravitational lensing simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atokiwaipmu/LensingSSC",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "healpy",
        "matplotlib",
        "astropy",
        "lenstools",
        "scipy",
        "h5py",
        "pyyaml",
        "classy",
        "pandas",
        "cobaya",
    ],
    include_package_data=True,
)