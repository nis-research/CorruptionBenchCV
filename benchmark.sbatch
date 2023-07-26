#!/bin/bash
#SBATCH -J test
#SBATCH -c 16
#SBATCH -p dmb
#SBATCH --gres=gpu:1
##SBATCH --nodelist=ctit092
## SBATCH --qos=research-deadline
#SBATCH --mail-type=END,FAIL 
#load modules
module load nvidia/cuda-11.6
# module load nvidia/cuda-11.3_cudnn-8.2

module load anaconda3/2022.05

#source activate
source activate pytorchenvi

if [ ! -d "/local/swang/ImageNet_C/brightness/" ]
then
cd /local/
mkdir swang/
cd swang/





cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_C/  .
# cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_C_bar/  .
# cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_3DCC/  .
# cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_P/  .


# cd ImageNet_3DCC/
# for corruption in near_focus far_focus fog_3d flash color_quant low_light xy_motion_blur z_motion_blur iso_noise bit_error h265_abr h265_crf
# do 
#     echo Unzipping $corruption
#     # wget https://datasets.epfl.ch/3dcc/imagenet_3dcc/$corruption.tar.gz  
#     tar -xvf $corruption.tar.gz
#     rm $corruption.tar.gz
# done

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

echo $SLURM_NODELIST
cd /home/wangs1/benchmark/code


# python main.py  --model ResNet18_DuBIN   --ckpt ckpts/ResNet18_AugMax_DuBIN/best_SA.pth    --dataset ImageNet_3DCC  --image_size 224 --data_path /local/swang/  >  IN_3DCC/ResNet18_AugMax_DuBIN.out

# python main.py  --model ResNet18_DuBIN_DeepAug   --ckpt ckpts/ResNet18_AugMax_DuBIN_DeepAug/best_SA.pth    --dataset ImageNet_3DCC  --image_size 224 --data_path /local/swang/  > IN_3DCC/ResNet18_AugMax_DuBIN_DeepAug.out

# python main.py  --model augmix_resnet50   --ckpt ckpts/augmix_resnet50_checkpoint.pth.tar     --dataset ImageNet_3DCC  --image_size 224 --data_path /local/swang/  >  IN_3DCC/augmix_resnet50.out

# python main.py  --model tf_efficientnet_l2_ns    --dataset ImageNet_3DCC  --image_size 224 --data_path /local/swang/  > IN_3DCC/tf_efficientnet_l2_ns.out

# python main.py  --model convnext_large_in22ft1k    --dataset ImageNet_C  --image_size 224 --data_path /local/swang/  > IN_C/convnext_large_in22ft1k.out


# python main.py  --model resnet50    --dataset ImageNet_C --image_size 224 --data_path /local/swang/  > IN_C/resnet50.out
# python main.py  --model densenet161    --dataset ImageNet_3DCC --image_size 224 --data_path /local/swang/  > IN_3DCC/densenet161.out
python main.py  --model timm/resnetv2_101x1_bit.goog_in21k_ft_in1k   --dataset ImageNet_C --image_size 224 --data_path /local/swang/  > IN_C/resnetv2_101x1_bitgoog_in21k_ft_in1k.out



# python main.py  --model tf_efficientnet_l2_ns    --dataset ImageNet_3DCC  --image_size 224 --data_path /local/swang/  > tf_efficientnet_l2_ns.out


# python main.py --ckpt /home/wangs1/Bias/results_imagenet10/imagenet10_30/resnet180\ /version_0/checkpoints/last.ckpt --ckpt_baseline /home/wangs1/Bias/results_imagenet10/imagenet10_00/resnet180\ /version_0/checkpoints/last.ckpt  --dataset ImageNet_C --image_size 224 --data_path /home/wangs1/datasets/  > summary_imagenetc30.out
# python main.py --ckpt /home/wangs1/Bias/results_cifar/90/resnet180\ /version_0/checkpoints/last.ckpt --ckpt_baseline /home/wangs1/HFC/results/cifar/resnet180\ /version_0/checkpoints/last.ckpt  --dataset cifar --image_size 32 --data_path /home/wangs1/datasets/  > summary_cifar90.out
# python main.py --ckpt /home/wangs1/Bias/results_cifar/50/resnet180\ /version_0/checkpoints/last.ckpt --ckpt_baseline /home/wangs1/HFC/results/cifar/resnet180\ /version_0/checkpoints/last.ckpt  --dataset cifar --image_size 32 --data_path /home/wangs1/datasets/  > summary_cifar50.out
# python main.py --ckpt /home/wangs1/Bias/results_cifar/30/resnet180\ /version_0/checkpoints/last.ckpt --ckpt_baseline /home/wangs1/HFC/results/cifar/resnet180\ /version_0/checkpoints/last.ckpt  --dataset cifar --image_size 32 --data_path /home/wangs1/datasets/  > summary_cifar30.out


cd /local/swang/

# rm -r  ImageNet_3DCC
 
conda deactivate