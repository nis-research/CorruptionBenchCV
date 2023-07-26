
import torch
from augmax.models.imagenet.resnet_DuBIN import ResNet18_DuBIN


model = ResNet18_DuBIN()
weight = torch.load('../Downloads/ResNet18_AugMax_DuBIN/best_SA.pth',map_location=torch.device('cpu'))

weight_new ={}
for k in weight:
    weight_new[k[7:]]=weight[k]

model.load_state_dict(weight_new)
model.eval()
