import torch
import torch.nn as nn
import torchvision.transforms as transforms



from pytorch_lightning.core.lightning import LightningModule
import torchmetrics

# from pytorch_lightning import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau




class Model(LightningModule):
    def __init__(self,backbone_model, decoder, lr,num_class,weight_alpha,dataset,image_size, band, masks, p , special=None):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.weight_alpha = weight_alpha
        self.dataset = dataset
        self.num_class = num_class
        self.image_size = image_size
        self.backbone_model = backbone_model
        self.decoder = decoder

        self.p = p
        self.masks = masks
        self.band = band
        self.special = special
    def forward(self, x):

        enc, prediction = self.backbone_model(x)

        if self.decoder is None:
            return enc, prediction

        # elif type(self.decoder) is not None:
        #     rct = self.decoder(out)
        #     return enc, rct, prediction

        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=1e-4)#torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=5e-04)
        # scheduler = StepLR(optimizer,step_size=20)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',verbose=True, factor=0.1)#, step_size=2, gamma=0.95)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()
        if self.decoder is None:
            _, y_hat = self(x)
            #print(y_hat)
            loss1 = criterion1(y_hat, y)
            loss = loss1
        elif type(self.decoder) is not None:
            _,rct, y_hat = self(x)
            loss1 = criterion1(y_hat, y)
            loss2 = criterion2(rct,x)
            loss = loss1 +self.weight_alpha*loss2
            self.log_dict({'train_reconstruction_loss': loss2}, on_epoch=True,on_step=True)
                    
        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict({'train_classification_loss': loss1}, on_epoch=True,on_step=True)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return {"loss": loss,'epoch_preds': predicted, 'epoch_targets': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()
        if self.decoder is None:
            _, y_hat = self(x)
            # print(y_hat)
            # print(y_hat.size())
            loss1 = criterion1(y_hat, y)
            self.val_loss = loss1

        elif type(self.decoder) is not None:
            _,rct, y_hat = self(x)
           
            loss1 = criterion1(y_hat, y)
            loss2 = criterion2(rct,x)
            self.val_loss = loss1 + self.weight_alpha*loss2
        
        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)

        return  {'epoch_preds': predicted, 'epoch_targets': y} #self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.decoder is None:
            _, y_hat = self(x)
            # print(y_hat.size())
        elif type(self.decoder) is not None:
            _,_, y_hat = self(x)
          
        _, predicted = torch.max(y_hat.data,1)
        
        return {'batch_preds': predicted, 'batch_targets': y}
        
    
    def test_step_end(self, output_results):
        
        self.test_acc(output_results['batch_preds'], output_results['batch_targets'])
        self.log_dict( {'test_acc': self.test_acc}, on_epoch=True,on_step=False)
        
    def training_epoch_end(self, output_results):
        # print(output_results)
        self.train_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"train_acc": self.train_acc}, on_epoch=True, on_step=False)

    def validation_epoch_end(self, output_results):
        # print(output_results)
        self.val_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"valid_acc": self.val_acc}, on_epoch=True, on_step=False)
        # print(acc)
        # return val_accuracy

    def setup(self, stage):

        if self.dataset == 'cifar':
            mean = [0.491400, 0.482158, 0.446531]
            std = [0.247032, 0.243485, 0.261588]
           
    
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
            data_train  = CIFAR_BP('../datasets/',train=True,band=self.band,transform=transform_train)
            data_test = CIFAR_BP('../datasets',train=False,band=self.band,transform=transform)

        
       
       
       
        # train/val split
        data_train2, data_val =  torch.utils.data.random_split(data_train, [int(len(data_train)*0.9), len(data_train)-int(len(data_train)*0.9)])

        # assign to use in dataloaders
        self.train_dataset = data_train2
        self.val_dataset = data_val
        self.test_dataset = data_test

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=16, shuffle=True)#,num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size= 16, shuffle=False)#,num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size= 16)#,num_workers=2)


