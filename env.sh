# Conda 環境を作成
conda create --prefix ./env python=3.8
conda activate ./env

# 必要なパッケージをインストール
conda install -c bccp nbodykit
conda install -c conda-forge mpi4py pyccl
python -m pip install cobaya --upgrade