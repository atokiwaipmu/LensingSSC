from setuptools import setup, find_packages

setup(
    name='lensingSSC',
    version='0.1',
    description='A package for analysing lensing simulations',
    author='Akira Tokiwa',
    author_email='akira.tokiwa@ipmu.jp',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyyaml',
        'healpy',
        'matplotlib',
        'classy',
        'lenstools',
        'astropy',
        'nbodykit',
        'pandas',
        'scipy',
    ],
    python_requires='>=3.6',
)

# Installation instructions

# To use this `setup.py`, save it in the root directory of your project.
# To install your package, run the following command in your terminal:
#
#   pip install .