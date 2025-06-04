from setuptools import setup, find_packages
import os

# Read the README file for a long description.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read core requirements from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# Define development and optional analysis dependencies
extras_require = {
    'dev': [
        'pytest>=6.0', 
        'cachetools>=4.0', 
        'flake8',
        'black',
        'mypy',
        'setuptools',
        'wheel',
        'twine'
    ],
    'analysis': [
        'pyyaml',
        'classy',
        'cobaya'
    ],
    'mysql': [
        'mysql-connector-python>=8.0.0'
    ]
}

setup(
    name="LensingSSC",
    version="0.1.0", 
    author="Your Name", # TODO: Update with actual author name
    author_email="your.email@example.com", # TODO: Update with actual author email
    description="A Python package for studying Super-Sample Covariance (SSC) effects in weak gravitational lensing simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atokiwaipmu/LensingSSC", 
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "notebooks", "LEGACY"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=install_requires, 
    extras_require=extras_require,
    include_package_data=True, 
    # entry_points={
    #     'console_scripts': [
    #         'lensing-ssc-preprocess=lensing_ssc.core.preprocessing.cli:main_preprocess',
    #     ],
    # },
)