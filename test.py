from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.datasets  import ImageFolder
from torchvision.transforms import transforms
import pandas as pd

from metric import CE, mCE, RelativeRobustness,ece_score,ECE, rCE, mrCE
from main import bs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_path = './datasets/' # change path later

def get_corruption_list(dataset):

    if dataset == 'ImageNet_C':
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

def test_corruption_severity(dataset, image_size,model_test, corruption,data_path, num_severity=5):
    corruption_severity_test = []

    
    
    if  dataset in ['ImageNet_C','ImageNet_C_bar','ImageNet_3DCC']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # transform=transforms.Compose([transforms.Resize((256,256)),
        #                               transforms.CenterCrop((image_size,image_size)),
        #                                transforms.ToTensor(),transforms.Normalize(mean,std)])
        transform=transforms.Compose([transforms.CenterCrop((image_size,image_size)),                    
                                       transforms.ToTensor(),transforms.Normalize(mean,std)])
   
    
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= bs, shuffle=False,num_workers=2)
            total = 0.0
            correct_test = 0.0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                outputs = model_test(images.float())
                _, predicted = torch.max(outputs.data,1)

              
                total += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            acc_test = float(100*correct_test/total)
            corruption_severity_test.append(acc_test)

    return corruption_severity_test 
                    


def test_ece_severity(dataset, image_size,model, corruption,data_path, num_severity=5):

    ece_severity_test = []
    

    if dataset in ['ImageNet_C','ImageNet_C_bar','ImageNet_3DCC']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform=transforms.Compose([transforms.CenterCrop((image_size,image_size)),
                                       transforms.ToTensor(),transforms.Normalize(mean,std)])
        
        # transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        #                                transforms.ToTensor(),transforms.Normalize(mean,std)])
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= bs, shuffle=False,num_workers=8)
            
            total = 0.0
            E = 0
            B = 0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                outputs = model(images.float())

                py_test = (outputs.cpu().detach().numpy())
                ytest = (labels.cpu().detach().numpy())

                total += labels.size(0)
        
                ece_test, Bm_test = ece_score(py_test,ytest)
                E += ece_test
                B += Bm_test

            ece = ECE(E,B)
            ece_severity_test.append(ece)


    return  ece_severity_test




def test_corruption(dataset, image_size,model_test,data_path):
    severity=5

    corruption_list = get_corruption_list(dataset)
    kong =  [0 for i in range(len(corruption_list))]
    d = {'Corruption': corruption_list, 'Acc_s1': kong,'Acc_s2': kong,'Acc_s3': kong,'Acc_s4': kong,'Acc_s5': kong,'ECE_s1': kong,'ECE_s2': kong,'ECE_s3': kong,'ECE_s4': kong,'ECE_s5': kong}
    d = pd.DataFrame(data=d)
    for corruption in corruption_list:
        acc_corruption_severity_test= test_corruption_severity(dataset, image_size,model_test, corruption,data_path)
        ece_severity_test = test_ece_severity(dataset, image_size,model_test, corruption,data_path)
        corruption_i = corruption_list.index(corruption)



        for s in range(severity):
            d.loc[corruption_i,'Acc_s'+str(s+1)] = acc_corruption_severity_test[s]
            d.loc[corruption_i,'ECE_s'+str(s+1)] = ece_severity_test[s]
       
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
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
            VideoFolder(root=data_path +"/ImageNet_P/" + perturbation,
                        transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

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

    return flip_prob(predictions),ranking_dist(ranks, mode='top5') # ,ranking_dist(ranks, mode='zipf')

def test_perturbations(model_test,difficulty,dataset, data_path):
  
    FPs = []
    T5Ds = []
    

    corruption_list = get_corruption_list(dataset)
    kong =  [0 for i in range(len(corruption_list))]
    d = {'Corruption': corruption_list, 'FP': kong,'T5D': kong}
    d = pd.DataFrame(data=d)
    for perturbation in corruption_list:
        FP_test,T5D_test= test_imagenet_p(model_test,perturbation,difficulty,data_path)
    
        perturbation_i = corruption_list.index(perturbation)

        d.loc[perturbation_i,'FP'] = FP_test
        d.loc[perturbation_i,'T5D'] = T5D_test

        FPs.append(FP_test)#/FP_baseline) 
        T5Ds.append(T5D_test)#/T5D_baseline)

    print(d)
    
    return d


def test_clean(model_test, data_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    transform=transforms.Compose([transforms.Resize((256,256)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.ToTensor(),transforms.Normalize(mean,std)])
    # transform=transforms.Compose([transforms.Resize((224,224)),                    
    #                                    transforms.ToTensor(),transforms.Normalize(mean,std)])

    data_test =  ImageFolder(data_path+'ImageNet/val/',transform=transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size= bs, shuffle=False,num_workers=2)
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

