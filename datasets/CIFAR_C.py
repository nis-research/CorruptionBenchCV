import os
import numpy as np
from torch.utils.data.dataset import Dataset

class CIFAR10_C(Dataset):
    def __init__(self, root_dir,corruption, transform = None):
        super(CIFAR10_C).__init__()

        self.labels_path = os.path.join(root_dir,'CIFAR-10-C','labels.npy')
        self.root_dir = os.path.join(root_dir,'CIFAR-10-C',corruption)+'.npy'
        # print(self.root_dir)
        self.transform = transform

        self.data = np.load(self.root_dir, allow_pickle=True)
        self.targets = np.load(self.labels_path, allow_pickle=True) 
        # self.data = self.data.transpose((0, 3, 1, 2))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        
        return img, target