#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=4000MB
#SBATCH --time=3-00:00
#SBATCH --account=def-jiayuan

class_name=$1

module load gcc/7.3.0 python/3.7 scipy-stack cuda/10
virtualenv .torch
source .torch/bin/activate
export FORCE_CUDA=1
pip install opencv-python
pip install torch==1.2.0
pip install https://github.com/pytorch/vision/archive/v0.4.0.tar.gz
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
 
mkdir -p models/${class_name}

python train.py ${class_name}
