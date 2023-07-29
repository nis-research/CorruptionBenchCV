# Benchmark Suite for Evaluating Corruption Robustness of Computer Vision Models
## Introduction
This is the benchmark framework used in the survey paper [The Robustness of Computer Vision Models against Common Corruptions: a Survey](https://arxiv.org/abs/2305.06024). It evaluates the corruption robustness of ImageNet-trained classifiers to benchmark datasets: [ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-C-bar](https://github.com/facebookresearch/augmentation-corruption), [ImageNet-P](https://github.com/hendrycks/robustness), [ImageNet-3DCC](https://github.com/EPFL-VILAB/3DCommonCorruptions).


<p align="center"><img src="figures/teaser.png" width="700"></p>
  
 ---
 ### Quick start
 Before running:
 
  * Download the datasets  
  * Install packages
   ```
   pip install -r requirements.txt
   ```
  
  
 Using your customize models:
 ``` 
 python main.py --ckpt model_path.ckpt --ckpt_baseline baseline_path.ckpt --dataset cifar 
 
 or using pre-trained models from timm
 
 ``` 
 python main.py --model resnet18  --dataset ImageNet_C_bar --image_size 224 --data_path /whereYouStoreImageNet_C_bar/
 ```

|Corruption|	Acc_s1|	Acc_s2	|Acc_s3	|Acc_s4| Acc_s5	|ECE_s1	|ECE_s2|	ECE_s3|	ECE_s4|	ECE_s5|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|blue_noise_sample	|   |   |   |   |   |   |   |   |   |   |


Outputs:



|Corruption|	Acc_s1|	Acc_s2	|Acc_s3	|Acc_s4| Acc_s5	|ECE_s1	|ECE_s2|	ECE_s3|	ECE_s4|	ECE_s5|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|blue_noise_sample	| | | | | | | | | | |
|brownish_noise|	 | | | | | | | | | |
|caustic_refraction	| | | | | | | | | | |
|checkerboard_cutout|	 | | | | | | | | | |
|cocentric_sine_waves	| | | | | | | | | | |
|inverse_sparkles	| | | | | | | | | | |
|perlin_noise|	 | | | | | | | | | |
|plasma_noise	| | | | | | | | | | |
|single_frequency_greyscale	| | | | | | | | | | |
|sparkles	| | | | | | | | | | |

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


  - metric.py
    - Accuracy (per corruption per severity) [For ImageNet - ALL]
    - Corruption Error (CE) [For ImageNet-C/-C-bar/-3DCC]
    - mCE [For ImageNet-C/-C-bar/-3DCC]
    - mFP [For ImageNet-P]
    - mT5D [For ImageNet-P]
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
 
