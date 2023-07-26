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
        transform=transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(),transforms.Normalize(mean,std)])
    
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= 32, shuffle=False,num_workers=2)

            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                _,outputs = model_test(images.float())
                _, predicted = torch.max(outputs.data,1)

                _,outputs_baseline = model_baseline(images.float())
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
        transform=transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(),transforms.Normalize(mean,std)])
    
        for severity in range(1,num_severity+1):
            data_test =  ImageFolder(data_path+dataset+'/'+corruption+'/'+str(severity),transform=transform)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size= 32, shuffle=False,num_workers=8)

            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)            
                _,outputs = model(images.float())

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

        print(corruption+': (from severity  1 to 5)')        
        print('Accuracy per severity: test model' + str(acc_corruption_severity_test))
        print('Accuracy per severity: basleine' +str(acc_corruption_severity_baseline))
   
        print('ECE per severity: test model' + str(ece_severity_test))
        print('ECE per severity: basleine' +str(ece_severity_baseline))

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
   
            
