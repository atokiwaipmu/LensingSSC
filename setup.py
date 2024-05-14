from setuptools import setup, find_packages
import os

setup(
    name='lensingSSC',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.8, <3.9',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'tqdm',
        'healpy',
        'bigfile',
        'matplotlib',
        'argparse',
        'json',
        'logging',
        'warnings',
        'mpi4py',
        'dataclasses'
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
    setup_requires=[
        'setuptools>=42',
        'wheel'
    ],
)