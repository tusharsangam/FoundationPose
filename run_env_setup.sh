#!/bin/bash
eval "$(conda shell.bash hook)"
echo "Creating conda env with packages"
conda env create --name foundationpose --file=environment.yml

# activate conda environment
conda activate foundationpose

echo "Installing pip dependencies"
python -m pip install -r requirements.txt

# # Install NVDiffRast
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# # Kaolin (Optional, needed if running model-free setup) [SKIP]
python -m pip install --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# # PyTorch3D
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

echo "Building necessary extention"
conda deactivate
conda activate foundationpose
# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

echo "Downloading demo_data"
mkdir weights
mkdir demo_data

# Use gdown to download the entire folder
URL='https://drive.usercontent.google.com/download?id=1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3&export=download&authuser=0&confirm=t&uuid=f22259a8-cea7-4857-a732-6bc32da57694&at=APZUnTVBKJmlMLpTulIdB49QySPP:1717458723690'
wget -O demo_data/mustard0.zip $URL
cd demo_data && unzip mustard0.zip
cd ..

echo "Please download the weights as per directed, can't automate !!"

python run_demo.py