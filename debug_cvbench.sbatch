#!/bin/bash
#SBATCH -J Training
#SBATCH -c 2
#SBATCH -p dmb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL 
#load modules
module load nvidia/cuda-11.7
module load nvidia/cuda-11.x_cudnn-8.6
module load anaconda3/2022.05

#source activate


echo $SLURM_NODELIST

cd  CorruptionBenchCV
source activate virtualenv

if [ ! -d "/local/swang/ImageNet_C/brightness/" ]
then
cd /local/
mkdir swang/
cd swang/
cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_C/  .
fi

if [ ! -d "/local/swang/ImageNet/val/" ]
then
cd /local/
mkdir swang/
cd swang/

mkdir ImageNet/
cd ImageNet/
cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet/imagenet-val.tar.gz  .
tar -xvf imagenet-val.tar.gz -o val
rm imagenet-val.tar.gz

fi

python main.py --model resnet18  --dataset ImageNet_C --image_size 224 --data_path /local/swang/
 
conda deactivate
