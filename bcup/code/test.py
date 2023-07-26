from datasets.CIFAR_C import CIFAR10_C
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.datasets  import ImageFolder
from torchvision.transforms import transforms
import pandas as pd

from metric import CE, mCE, RelativeRobustness,ece_score,ECE

import sys
sys.path.append("/home/wangs1/benchmark/code")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_path = './datasets/' # change path later

def get_corruption_list(dataset):

    if dataset == 'cifar':
    
        Corruptions_list = ['brightness','contrast','defocus_blur','elastic_transform',
                'fog','frost','gaussian_blur','gaussian_noise','glass_blur',
                  'impulse_noise','jpeg_compression','motion_blur','pixelate','saturate',
                  'shot_noise','snow','spatter','speckle_noise','zoom_blur']
    elif dataset == 'ImageNet_C':
    
        Corruptions_list = ['brightness','contrast','defocus_blur','elastic_transform',
                'fog','frost','gaussian_noise','glass_blur',
                  'impulse_noise','jpeg_compression','motion_blur','pixelate',
                  'shot_noise','snow', 'zoom_blur']
    elif dataset == 'ImageNet_C_bar':
        Corruptions_list = ['blue_noise_sample','brownish_noise','caustic_refraction','checkerboard_cutout',
                            'cocentric_sine_waves','inverse_sparkles','perlin_noise','plasma_noise',
                            'single_frequency_greyscale','sparkles']
    elif dataset == 'ImageNet_3DCC':
        Corruptions_list = ['near_focus', 'far_focus', 'fog_3d', 'flash', 'color_quant', 'low_light',
                             'xy_motion_blur', 'z_motion_blur', 'iso_noise', 'bit_error', 'h265_abr']
    elif dataset == 'ImageNet_P':
        Corruptions_list = ['brightness','gaussian_noise', 'motion_blur','rotate','scale','shot_noise','snow', 'tilt', 'translate',  'zoom_blur'
                                                        #  ',spatter',   'speckle_noise', 'gaussian_blur',  'shear'
                             ]

    
    return Corruptions_list

def test_corruption_severity(dataset, image_size,model_test, model_baseline, corruption,data_path, num_severity=5):
    corruption_severity_test = []
    corruption_severity_baseline = []
    # ece_severity_test = []
    # ece_severity_baseline = []

    total = 0.0
    correct_test = 0.0
    correct_baseline = 0.0
    # py_test = []
    # ytest = []
    # py_baseline = []

    if dataset == 'cifar':
    
        mean = [0.491400, 0.482158, 0.446531]
        std = [0.247032, 0.243485, 0.261588]

        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

        
        testset = CIFAR10_C(data_path,corruption,transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                        batch_size=100, 
                                        shuffle=False)
        
        
        
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            _,outputs = model_test(images.float())
            _, predicted = torch.max(outputs.data,1)
            
            # py_test.append(outputs)
            # ytest.append(labels)


            _,outputs_baseline = model_baseline(images.float())
            _, predicted_baseline = torch.max(outputs_baseline.data,1)
            # py_baseline.append(outputs_baseline)


            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            correct_baseline += (predicted_baseline == labels).sum().item()

            if total == 10000  :

                acc_test = float(100*correct_test/total)
                corruption_severity_test.append(acc_test)
                # ece_test = ece_score(py_test,ytest)
                # ece_severity_test.append(ece_test)

                acc_baseline = float(100*correct_baseline/total)
                corruption_severity_baseline.append(acc_baseline)
                # ece_baseline = ece_score(py_baseline,ytest)
                # ece_severity_baseline.append(ece_baseline)

                correct_test = 0.0
                correct_baseline = 0.0
                total = 0.0

      

   

    elif dataset in ['ImageNet_C','ImageNet_C_bar','ImageNet_3DCC']:
        mean = [0.479838, 0.470448, 0.429404]
        std = [0.258143, 0.252662, 0.272406]
        
        transform=transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((image_size,image_size)),
                                       transforms.ToTensor(),transforms.Normalize(mean,std)])
        # transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        #                                transforms.ToTensor(),transforms.Normalize(mean,std)])
    
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= 32, shuffle=False,num_workers=2)

            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                outputs = model_test(images.float())
                _, predicted = torch.max(outputs.data,1)

                outputs_baseline = model_baseline(images.float())
                _, predicted_baseline = torch.max(outputs_baseline.data,1)
              
                total += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                correct_baseline += (predicted_baseline == labels).sum().item()
           
            acc_test = float(100*correct_test/total)
            corruption_severity_test.append(acc_test)

            acc_baseline = float(100*correct_baseline/total)
            corruption_severity_baseline.append(acc_baseline)
          
            correct_test = 0.0
            correct_baseline = 0.0

    return corruption_severity_test, corruption_severity_baseline 
                    


def test_ece_severity(dataset, image_size,model, corruption,data_path, num_severity=5):

    ece_severity_test = []
    total = 0.0
    E = 0
    B = 0
   
     

    if dataset == 'cifar':
    
        mean = [0.491400, 0.482158, 0.446531]
        std = [0.247032, 0.243485, 0.261588]

        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

        
        testset = CIFAR10_C(data_path,corruption,transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                        batch_size=100, 
                                        shuffle=False)
        
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            _,outputs = model(images.float())
            
            py_test = (outputs.cpu().detach().numpy())
            ytest = (labels.cpu().detach().numpy())


            total += labels.size(0)

            ece_test, Bm_test = ece_score(py_test,ytest)
            E += ece_test
            B += Bm_test
            
          

            if total == 10000  :
                ece = ECE(E,B)
                ece_severity_test.append(ece)

                total = 0.0
                E = 0
                B = 0
               

    elif dataset in ['ImageNet_C','ImageNet_C_bar','ImageNet_3DCC']:
        mean = [0.479838, 0.470448, 0.429404]
        std = [0.258143, 0.252662, 0.272406]
        transform=transforms.Compose([transforms.Resize((image_size,image_size)),
                                       transforms.ToTensor(),transforms.Normalize(mean,std)])
        
        # transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        #                                transforms.ToTensor(),transforms.Normalize(mean,std)])
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= 1, shuffle=False,num_workers=8)

            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                outputs = model(images.float())

                py_test = (outputs.cpu().detach().numpy())
                ytest = (labels.cpu().detach().numpy())

    

                total += labels.size(0)
                E = 0
                B = 0
        
                ece_test, Bm_test = ece_score(py_test,ytest)
                E += ece_test
                B += Bm_test
                py_test = []
                ytest = []

            ece = ECE(E,B)
            ece_severity_test.append(ece)
            py_test = []
            ytest = []

    return  ece_severity_test




def test_corruption(dataset, image_size,model_test,model_baseline,data_path):
    acc_all_corruptions = []
    acc_all_corruptions_baseline = []
    ce_all_corruptions = []
    severity=5

    corruption_list = get_corruption_list(dataset)
    kong =  [0 for i in range(len(corruption_list))]
    d = {'Corruption': corruption_list, 'Acc_s1': kong,'Acc_s2': kong,'Acc_s3': kong,'Acc_s4': kong,'Acc_s5': kong,'ECE_s1': kong,'ECE_s2': kong,'ECE_s3': kong,'ECE_s4': kong,'ECE_s5': kong}
    # print(d)
    d = pd.DataFrame(data=d)
    for corruption in corruption_list:
        acc_corruption_severity_test,acc_corruption_severity_baseline= test_corruption_severity(dataset, image_size,model_test,model_baseline, corruption,data_path)
        ece_severity_test = test_ece_severity(dataset, image_size,model_test, corruption,data_path)
        ece_severity_baseline = test_ece_severity(dataset, image_size,model_baseline, corruption,data_path)
        acc_corruption =  sum(acc_corruption_severity_test)   
        ce_corruption_test = [100 - r for r in acc_corruption_severity_test] 
        ce_corruption_baseline = [100 -r for r in acc_corruption_severity_baseline]
        corruption_i = corruption_list.index(corruption)

        # print(corruption+': (from severity  1 to 5)')        
        # print('Accuracy per severity: test model' + str(acc_corruption_severity_test))
        # print('Accuracy per severity: basleine' +str(acc_corruption_severity_baseline))
   
        # print('ECE per severity: test model' + str(ece_severity_test))
        # print('ECE per severity: basleine' +str(ece_severity_baseline))

        for s in range(severity):
            d.loc[corruption_i,'Acc_s'+str(s+1)] = acc_corruption_severity_test[s]
            d.loc[corruption_i,'ECE_s'+str(s+1)] = ece_severity_test[s]
        CE_corruption = CE(ce_corruption_test,ce_corruption_baseline)
        ce_all_corruptions.append(CE_corruption)
        

        acc_all_corruptions.append(acc_corruption/5.0)
        acc_all_corruptions_baseline.append(sum(acc_corruption_severity_baseline)/5.0)
    avg_acc = sum(acc_all_corruptions)/len(acc_all_corruptions)
    avg_acc_baseline = sum(acc_all_corruptions_baseline)/len(acc_all_corruptions_baseline)

    print('Average accuracy test: %.2f' %avg_acc)
    print('Average accuracy baseline: %.2f' %avg_acc_baseline)
    relR = RelativeRobustness(avg_acc,avg_acc_baseline)
    print('Relative Robustness %.3f' %relR)
    mCE_final = mCE(ce_all_corruptions)
    print('mCE: %.2f' % mCE_final)
    print(d)
    
    return d
   




# ----------------for ImageeNet-P -------------
import numpy as np



            
def test_imagenet_p(model,perturbation,difficulty,data_path):
    from tqdm import tqdm
    import torchvision.transforms as trn
    from utils.video_loader import VideoFolder
    import numpy as np
    from scipy.stats import rankdata
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    identity = np.asarray(range(1, 1001))
    cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (999 - 5)))
    recip = 1./identity


    def dist(sigma, mode='top5'):
        if mode == 'top5':
            return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
        elif mode == 'zipf':
            return np.sum(np.abs(recip - recip[sigma-1])*recip)


    def ranking_dist(ranks, noise_perturbation=True if 'noise' in perturbation else False, mode='top5'):
        result = 0
        step_size = 1 if noise_perturbation else difficulty

        for vid_ranks in ranks:
            result_for_vid = []

            for i in range(step_size):
                perm1 = vid_ranks[i]
                perm1_inv = np.argsort(perm1)

                for rank in vid_ranks[i::step_size][1:]:
                    perm2 = rank
                    result_for_vid.append(dist(perm2[perm1_inv], mode))
                    if not noise_perturbation:
                        perm1 = perm2
                        perm1_inv = np.argsort(perm1)

            result += np.mean(result_for_vid) / len(ranks)

        return result


    def flip_prob(predictions, noise_perturbation=True if 'noise' in perturbation else False):
        result = 0
        step_size = 1 if noise_perturbation else difficulty

        for vid_preds in predictions:
            result_for_vid = []

            for i in range(step_size):
                prev_pred = vid_preds[i]

                for pred in vid_preds[i::step_size][1:]:
                    result_for_vid.append(int(prev_pred != pred))
                    if not noise_perturbation: prev_pred = pred

            result += np.mean(result_for_vid) / len(predictions)

        return result

    if difficulty > 1 and 'noise' in perturbation:
        loader = torch.utils.data.DataLoader(
            VideoFolder(root=data_path + "/ImageNet_P/" +
                            perturbation + '_' + str(difficulty),
                        transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
            batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
            VideoFolder(root=data_path +"/ImageNet_P/" + perturbation,
                        transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
            batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    predictions, ranks = [], []
    with torch.no_grad():

        for data, target in loader:
            num_vids = data.size(0)
            data = data.view(-1,3,224,224).cuda()

            output = model(data)

            for vid in output.view(num_vids, -1, 1000):
                predictions.append(vid.argmax(1).to('cpu').numpy())
                ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])


    ranks = np.asarray(ranks)
 
    # print('Computing Metrics\n')

    # print('Flipping Prob\t{:.5f}'.format(flip_prob(predictions)))
    # print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, mode='top5')))
    # print('Zipf Distance\t{:.5f}'.format(ranking_dist(ranks, mode='zipf')))

    return flip_prob(predictions),ranking_dist(ranks, mode='top5') # ,ranking_dist(ranks, mode='zipf')

def test_perturbations(model_test,model_baseline,difficulty,dataset, data_path):
  
    FPs = []
    T5Ds = []
    

    corruption_list = get_corruption_list(dataset)
    kong =  [0 for i in range(len(corruption_list))]
    d = {'Corruption': corruption_list, 'FP': kong,'T5D': kong}
    d = pd.DataFrame(data=d)
    for perturbation in corruption_list:
        FP_test,T5D_test= test_imagenet_p(model_test,perturbation,difficulty,data_path)
        FP_baseline,T5D_baseline= test_imagenet_p(model_baseline,perturbation,difficulty,data_path)
    
        perturbation_i = corruption_list.index(perturbation)

        d.loc[perturbation_i,'FP'] = FP_test
        d.loc[perturbation_i,'T5D'] = T5D_test

        FPs.append(FP_test/FP_baseline) 
        T5Ds.append(T5D_test/T5D_baseline)

        

       
    mFP = sum(FPs)/len(FPs)*100.0
    mT5D = sum(T5Ds)/len(T5Ds)*100.0

    print('mFP: %.2f' %mFP)
    print('mT5D: %.2f' %mT5D)
  
    print(d)
    
    return d

def test_clean(model_test, data_path):
    mean = [0.479838, 0.470448, 0.429404]
    std = [0.258143, 0.252662, 0.272406]
        
    transform=transforms.Compose([transforms.Resize((256,256)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.ToTensor(),transforms.Normalize(mean,std)])

    data_test =  ImageFolder(data_path+'ImageNet/val/',transform=transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size= 16, shuffle=False,num_workers=2)
    total = 0.0
    correct_test = 0.0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)            
        outputs = model_test(images.float())
        _, predicted = torch.max(outputs.data,1)

        total += labels.size(0)
        correct_test += (predicted == labels).sum().item()
    
    acc_test = float(100*correct_test/total)

    return acc_test

