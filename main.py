import torch
import argparse
import numpy as np

import os
import timm

#Add here necessary libraries to load your custom model




global bs 
bs = 8
   


def main(args):
    print(torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if model name from TIMM contains a "/"
    model_name_fileout = args.model.replace("/", "-")
    #model_name_fileout = model_name_fileout.replace(".", "")
    if args.ckpt == None:
        model_test = timm.create_model(args.model, pretrained=True).to(device).eval()
        model_baseline = timm.create_model('resnet18', pretrained=True).to(device).eval()
        from test import test_corruption, test_perturbations
    else:
        if args.model == 'ResNet18_DuBIN' or args.model == 'ResNet18_DuBIN_DeepAug':
            model_test = ResNet18_DuBIN().to(device)
            weight = torch.load(args.ckpt)
            weight_new = {}
            for k in weight:
                weight_new[k[7:]] = weight[k]

            model_test.load_state_dict(weight_new)
            model_test.eval()
            from test import test_corruption, test_perturbations

        elif args.model == 'augmix_resnet50':
            from torchvision import models
            model_test = models.resnet50().to(device)
            weight = torch.load(args.ckpt)['state_dict']
            # print(weight.keys())
            weight_new = {}
            for k in weight:
                weight_new[k[7:]] = weight[k]

            model_test.load_state_dict(weight_new)
            model_test.eval()
            from test import test_corruption, test_perturbations
        else:
            model_test = Model.load_from_checkpoint(args.ckpt).to(device).eval()
            from test_local import test_corruption

        model_baseline = timm.create_model('resnet18', pretrained=True).to(device).eval()
        # Model.load_from_checkpoint(args.ckpt_baseline).to(device).eval()

    if args.dataset == 'ImageNet_all':
        datasets = {'ImageNet_all': ['ImageNet_C', 'ImageNet_3DCC', 'ImageNet_C_bar']}

        for dataset in datasets[args.dataset]:
            d = test_corruption(dataset, args.image_size, model_test, model_baseline, args.data_path)
            d.to_csv('summary' + dataset + '_' + model_name_fileout + '.csv')

        d = test_perturbations(model_test, model_baseline, args.difficulty, 'ImageNet_P', args.data_path)
        d.to_csv('summaryImageNet_P_' + model_name_fileout + '.csv')

    elif args.dataset == 'ImageNet_P':
        if not os.path.exists('results/'):
            os.makedirs('results/')

        if not os.path.exists('results/summary' + args.dataset + '_' + model_name_fileout + '.csv'):
            d_test = test_perturbations(model_test, args.difficulty, args.dataset, args.data_path)
            d_test.to_csv('results/summary' + args.dataset + '_' + model_name_fileout + '.csv')

        if not os.path.exists('results/summary' + args.dataset + '_resnet18.csv'):
            d_baseline = test_perturbations(model_baseline, args.difficulty, args.dataset, args.data_path)
            d_baseline.to_csv('results/summary' + args.dataset + '_resnet18.csv')

        from metric import get_mfp_mt5d
        mfp, mt5d = get_mfp_mt5d('results/summary' + args.dataset + '_' + model_name_fileout + '.csv',
                                 'results/summary' + args.dataset + '_resnet18.csv')
        print('mFP: %.2f , mT5D: %.2f' % (mfp, mt5d))

    elif args.dataset == 'ImageNet':
        from test import test_clean
        acc = test_clean(model_test, args.data_path)
        print(acc)

    else:
        if torch.cuda.device_count() > 1:
            model_test = torch.nn.DataParallel(model_test, device_ids=list(range(torch.cuda.device_count())))
            model_baseline = torch.nn.DataParallel(model_baseline, device_ids=list(range(torch.cuda.device_count())))

        if not os.path.exists('results/'):
            os.makedirs('results/')

        if not os.path.exists('results/summary' + args.dataset + '_' + model_name_fileout + '.csv'):
            d_test = test_corruption(args.dataset, args.image_size, model_test, args.data_path)
            d_test.to_csv('results/summary' + args.dataset + '_' + model_name_fileout + '.csv')
        if not os.path.exists('results/summary' + args.dataset + '_resnet18.csv'):
            d_baseline = test_corruption(args.dataset, args.image_size, model_baseline, args.data_path)
            d_baseline.to_csv('results/summary' + args.dataset + '_resnet18.csv')

        from test import test_clean
        from metric import get_rCE, get_mCE, get_acc, RelativeRobustness, get_ece

        if args.test_clean_acc is None:
            acc_test = test_clean(model_test, args.data_path)
        else:
            acc_test = args.test_clean_acc

        if args.baseline_clean_acc is None:
            acc_baseline = test_clean(model_baseline, args.data_path)
        else:
            acc_baseline = args.baseline_clean_acc

        print("Clean accuracy")
        print('Test model-' + model_name_fileout + ': %.2f' % acc_test)
        print('Base model-ResNet18: %.2f' % acc_baseline + '\n')

        result_acc = get_acc('results/summary' + args.dataset + '_' + model_name_fileout + '.csv')
        print('Robust Accuracy')
        print('Test model-' + args.model + ': %.2f' % result_acc)
        result_acc_base = get_acc('results/summary' + args.dataset + '_resnet18.csv')
        print('Base model-ResNet18: %.2f' % result_acc_base)
        relative_acc = RelativeRobustness(result_acc, result_acc_base)
        print('Relative Robustness: %.2f \n' % relative_acc)

        result_rCE = get_rCE('results/summary' + args.dataset + '_' + model_name_fileout + '.csv',
                             'results/summary' + args.dataset + '_resnet18.csv', 100 - acc_test, 100 - acc_baseline)
        print('relative mCE: %.2f' % result_rCE)
        result_mCE = get_mCE('results/summary' + args.dataset + '_' +model_name_fileout + '.csv',
                             'results/summary' + args.dataset + '_resnet18.csv')
        print('mCE: %.2f \n' % result_mCE)

        result_ECE = get_ece('results/summary' + args.dataset + '_' + model_name_fileout + '.csv',
                             'results/summary' + args.dataset + '_resnet18.csv', )
        print('Average and normalized ECE: %.3f' % result_ECE)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--model', type=str, default= None,
                    help='model with pretrained weights available in timm')
    parser.add_argument('--test_clean_acc', type=int, default= None,
                    help='test accuracy of the tested model on clean test set')
    parser.add_argument('--baseline_clean_acc', type=int, default= None,
                    help='test accuracy of the baseline model on clean test set')
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
    parser.add_argument('--difficulty', '-d', type=int, default=1, choices=[1, 2, 3])
    

   
    args = parser.parse_args()
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    main(args)