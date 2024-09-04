
# Create a conda environment with the required packages
conda create --prefix ./env python=3.8
conda activate ./env
conda install -c bccp nbodykit
conda install -c conda-forge mpi4py
python -m pip install cobaya --upgrade
conda install -c conda-forge pyccl

# Install the required GCC compiler, following the instructions here:https://hpc.oarc.ucla.edu/docs/swe/building-gcc.html
# Then, add the following lines to your .bashrc file: