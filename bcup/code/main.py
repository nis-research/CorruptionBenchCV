import torch
import argparse
import numpy as np


import timm
#Add here necessary libraries to load your custom model
from train import Model
from augmax.models.imagenet.resnet_DuBIN import ResNet18_DuBIN




   

def main(args):


    print(torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.ckpt == None:
        model_test = timm.create_model(args.model, pretrained=True).to(device).eval()
        model_baseline = timm.create_model('resnet18', pretrained=True).to(device).eval()
        from test import test_corruption,test_perturbations 
    else:
        if args.model == 'ResNet18_DuBIN' or args.model == 'ResNet18_DuBIN_DeepAug':
            model_test = ResNet18_DuBIN().to(device)
            weight = torch.load(args.ckpt)
            weight_new ={}
            for k in weight:
                weight_new[k[7:]]=weight[k]

            model_test.load_state_dict(weight_new)
            model_test.eval()
            from test import test_corruption,test_perturbations
            
        elif args.model == 'augmix_resnet50':
            from torchvision import models 
            model_test = models.resnet50().to(device)
            weight = torch.load(args.ckpt)['state_dict']
            # print(weight.keys())
            weight_new ={}
            for k in weight:
                weight_new[k[7:]]=weight[k]


            model_test.load_state_dict(weight_new)
            model_test.eval()
            from test import test_corruption,test_perturbations
        else:
            model_test = Model.load_from_checkpoint(args.ckpt).to(device).eval()
            from test_local import test_corruption
        # print(model_to_test)
        model_baseline = timm.create_model('resnet18', pretrained=True).to(device).eval()
        # Model.load_from_checkpoint(args.ckpt_baseline).to(device).eval()
            

    if args.dataset == 'ImageNet_all':
        datasets = {'ImageNet_all':['ImageNet_C','ImageNet_3DCC','ImageNet_C_bar']}

        for dataset in datasets[args.dataset]:
            d =test_corruption(dataset,args.image_size, model_test,model_baseline,args.data_path)
            d.to_csv('summary'+dataset+'_'+args.model+'.csv')


        d = test_perturbations(model_test,model_baseline,args.difficulty,'ImageNet_P', args.data_path)
        d.to_csv('summaryImageNet_P_'+args.model+'.csv')

    elif args.dataset == 'ImageNet_P':
        d = test_perturbations(model_test,model_baseline,args.difficulty,args.dataset, args.data_path)
        d.to_csv('summary'+args.dataset+'_'+args.model+'.csv')
    elif args.dataset == 'ImageNet':
        from test import test_clean
        acc = test_clean(model_test, args.data_path)
        print(acc)
      
    else:
        if torch.cuda.device_count() > 1:
            model_test = torch.nn.DataParallel(model_test, device_ids=list(range(torch.cuda.device_count())))
            model_baseline = torch.nn.DataParallel(model_baseline, device_ids=list(range(torch.cuda.device_count())))
        d = test_corruption(args.dataset,args.image_size, model_test,model_baseline,args.data_path)
        d.to_csv('summary'+args.dataset+'_'+args.model+'.csv')

 
    

    
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--model', type=str, default= None,
                    help='model with pretrained weights available in timm')
    parser.add_argument('--ckpt', type=str,default=None,
                    help='checkpoint of a model')
    parser.add_argument('--ckpt_baseline', type=str,
                    help='checkpoint of a baseline model')
    parser.add_argument('--dataset', type=str, default='cifar',
                    help='dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/',
                    help='data_path')
    parser.add_argument('--image_size', type=int, default= 32,
                    help='size of images in dataset')
    # parser.add_argument('--perturbation', '-p', default='brightness', type=str,
    #                 choices=['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
    #                          'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale',
    #                          'speckle_noise', 'gaussian_blur', 'snow', 'shear'])
    parser.add_argument('--difficulty', '-d', type=int, default=1, choices=[1, 2, 3])
    

   
    args = parser.parse_args()
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    main(args)