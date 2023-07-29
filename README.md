# Benchmark Suite for Evaluating Corruption Robustness of Computer Vision Models
## Introduction
This is the benchmark framework used in the survey paper [The Robustness of Computer Vision Models against Common Corruptions: a Survey](https://arxiv.org/abs/2305.06024). It evaluates the corruption robustness of ImageNet-trained classifiers to benchmark datasets: [ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-C-bar](https://github.com/facebookresearch/augmentation-corruption), [ImageNet-P](https://github.com/hendrycks/robustness), [ImageNet-3DCC](https://github.com/EPFL-VILAB/3DCommonCorruptions).


<p align="center"><img src="figures/teaser.png" width="700"></p>
  
 ---
 ### Quick start
1. Before running:
 
  * Download the datasets  
  * Install packages
   ```
   pip install -r requirements.txt
   ```
  
  
2. Evaluating pre-trained models from [timm](https://huggingface.co/models?sort=downloads&search=bit)
 
 ``` 
 python main.py --model resnet18  --dataset ImageNet_C_bar --image_size 224 --data_path /whereYouStoreImageNet_C_bar/
 ```

Output: a csv file with the following structure, recording the accuracy and ECE per corruption per severity. The overal results (e.g. clean accuracy, robust accuracy, relative robustness, relative mCE, mCE, mFP, and mT5D) will be printed. You can also use the csv file to compute the values of the above metrics.
 


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
 *  Change this parameter to select the pretrained model available in [timm](https://huggingface.co/models?sort=downloads&search=bit)  '--model'
 *  For testing on all benchmark datasets, it has high requirement of storage. Thus, we suggest testing on benchmark datasets one by one. 

3. Comparing robustness among different backbones
  
