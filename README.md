# BenchmarkSuiteRobustDL
A benchmark suite for testing the robustness of deep learning models towards common image corruptions
---

## Introduction
 
### Folder structure 

  - Datasets
    - [CIFAR10-C](https://zenodo.org/record/2535967#.Y-J1AxXMKUl)
    - [ImageNet-C](https://github.com/hendrycks/robustness)
    - [ImageNet-C-bar](https://github.com/facebookresearch/augmentation-corruption)
    - [ImageNet-P](https://github.com/hendrycks/robustness)
    - [ImageNet-3DCC](https://github.com/EPFL-VILAB/3DCommonCorruptions)
    - [ImageNet-D](not publically available yet)
    - [ImageNet-Cartoon](https://zenodo.org/record/6801109#.Y-J1yBXMKUk)
    - [ImageNet-Drawing](https://zenodo.org/record/6801109#.Y-J1yBXMKUk)
  - metric.py
    - Accuracy (per corruption per severity) [For ImageNet - ALL]
    - Corruption Error (CE) [For ImageNet-C/-C-bar/-3DCC]
    - mCE [For ImageNet-C/-C-bar/-3DCC]
    - Relative robustness [For ImageNet - ALL]
    - Expected callibration error (ECE) [For ImageNet - ALL]
  - test.py
    - load a dataset
    - output test results
  - main.py
    - load a model (+ a baseline model)
    - test a model on all datasets
    - output a summary of test results
  - notebook
    - plot results based on summary
  
 ---
 ### Quick start
 Before running:
 
  * Download the datasets  
  * Install libraries neccesary to load the model (e.g., folders *backbone* and *blocks*)
  * Other libraries to install 
   ```
   pip install -r requirements.txt
   ```
  
  
 Using your customize models:
 ``` 
 python main.py --ckpt model_path.ckpt --ckpt_baseline baseline_path.ckpt --dataset cifar > summary.out
 ```
 
 or using pre-trained models from timm
 
 ``` 
 python main.py --model resnet18  --dataset ImageNet_C_bar --image_size 224 --data_path /whereYouStoreImageNet_C_bar/  > summary.out
 ```
 ---
 Notice: 
 * It is important to define the file name 'summary.out', because it saves the printed results of the datasets (e.g. mCE, mFP, ......).
 *  Change this parameter to select the pretrained model  '--model'
 * Copy datasets first, e.g. 
 ``` 
 cp -r /deepstore/datasets/dmb/ComputerVision/nis-data/shunxin/ImageNet_P/  /local/swang/
 ```
 ---
 
 For testing on all benchmark datasets, it has high requirement of storage. Thus, we suggest testing on benchmark datasets one by one. 
 
