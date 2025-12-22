import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        convnext = models.efficientnet_v2_s(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(convnext.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x    

class EncoderMobile(nn.Module):
    def __init__(self):
        super(EncoderMobile, self).__init__()
        convnext = models.mobilenet_v3_small(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(convnext.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x    



class byol_encoder():

    def __init__(self,model_path='79_student_encoder_512_90_wd5_lr_7_contrast_1augment_epoch_43.pth'):
        self.model=Encoder()
        self.model.load_state_dict(torch.load(model_path))
        self.model=self.model.to("cuda").eval()
    def __call__(self,image,transform=None):
        if transform is not None:
            image=transform(image)
        with torch.no_grad():
            embed = self.model(image.cuda())
        return embed

class mobile_encoder():

    def __init__(self,model_path='/media/pixo/Volume2/kohler_coating/paper_finder/checkpoints/79_student_encoder_512_90_wd5_lr_7_contrast_1augment_epoch_43.pth'):
        self.model=EncoderMobile().eval().cuda()
    
    def __call__(self,image,transform=None):
        if transform is not None:
            image=transform(image)
        with torch.no_grad():
            embed = self.model(image.cuda())
        return embed


class SSCD_model():

    def __init__(self,model_path='sscd_imagenet_mixup.torchscript.pt'):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        self.small_288 = transforms.Compose([
            transforms.Resize(288),
            #transforms.ToTensor(),
            normalize,
        ])
        self.model = torch.jit.load(model_path).cuda().eval()


    def __call__(self,image,transform=None):
        #if len(image.shape) == 4:
        #    image = image.squeeze()
        if transform is None:
            with torch.no_grad():
                #img = Image.open(i).convert('RGB')
                #batch = self.small_288(image).unsqueeze(0)
                #embed = self.model(batch.cuda())   
                embed = self.model(image.cuda())   
        return embed