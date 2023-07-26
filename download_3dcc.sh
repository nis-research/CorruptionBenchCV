##!/usr/bin/env bash
# mkdir ImageNet_3DCC
# cd ImageNet_3DCC
cd /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_3DCC
for corruption in near_focus far_focus fog_3d flash color_quant low_light xy_motion_blur z_motion_blur iso_noise bit_error h265_abr h265_crf
do 
    echo downloading $corruption
    wget https://datasets.epfl.ch/3dcc/imagenet_3dcc/$corruption.tar.gz 
    # tar $corruption.tar.gz
done