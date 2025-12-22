from lightglue import LightGlue, DISK
from superpoint_modified import SuperPointPixo
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torchvision
from torch.nn import functional as F
import numpy as np
import torch
import torchvision.transforms.functional as fcx

from safetensors.torch import load_file,save_file
from torchvision import transforms as v2
class decision_maker():

    def __init__(self,extractor_model='superpoint',keypoints=128):

        if extractor_model=='superpoint':
            self.extractor = SuperPointPixo(max_num_keypoints=keypoints).eval().cuda()  # load the extractor
        elif extractor_model=='disk':
            self.extractor = DISK(max_num_keypoints=keypoints,weights="epipolar").eval().cuda()  # load the extractor
        elif extractor_model=='xfeat':
            self.extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = keypoints)
        self.resizer=v2.Resize((1024,768))
        self.extractor_model=extractor_model
    def __call__(self,img,height=1024,width=768,padding=True,save_path=None,batch_size=4,rotate=False,num_patch=1):
        
        if img.max()>1:
            img=(img.float())/255
        if img.shape[1] < img.shape[2]:  # img.shape[2] is W, img.shape[1] is H
            # Transpose the dimensions to make width the last dimension
            img = fcx.hflip(img.permute(0,2,1))
        image0=img.cuda()

        _,h,w=image0.shape

        if padding:
            paddingx = (0,int(np.ceil(w/3)*3-w),int(np.ceil(h/3)*3-h),0)
            image0=F.pad(image0,paddingx)
        _,h,w=image0.shape

        if rotate:
            images_rot2=fcx.rotate(image0,180)
            image0=torch.stack((image0,images_rot2),dim=0)
        #images=self.resizer(images)
        images=image0.split(int(w/num_patch),-1)
        images=[image.split(int(h/num_patch),-2) for image in images]
      
        images=list(sum(images, ()))
        if rotate:
            images+=image0.split(1,dim=0)
        else:
            images.append(image0)
        
        images_=[]
        for im in images:
            if im.shape[0] == 2:
                images_.append(im[0].unsqueeze(0))
                images_.append(im[1].unsqueeze(0))
            else:
                images_.append(im[0].unsqueeze(0))
        images_ = [i for i in images_ if i.shape[2]>1 and i.shape[3]>1]
        images=[self.resizer(image) for image in images]
        if rotate:
            images=torch.cat(images,0)
        else:
            images=torch.stack(images,0)
        ft=[]
        for i in range(0,len(images_),batch_size):
            if self.extractor_model=='xfeat':
                with torch.no_grad():
                    ft+=(self.extractor.detectAndComputeDense(images[i:i+batch_size])['descriptors'])
            else:
                with torch.no_grad():
                    #ft+=(self.extractor.extract_batch(images[i:i+batch_size])['descriptors'])
                    ft+=(self.extractor.extract(images_[i:i+batch_size][0].float())['descriptors'])
        max_len = max(t.shape[0] for t in ft)
        ft=[F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in ft]
        array=F.normalize(torch.stack(ft,dim=0),p=2,dim=-1)
        if save_path is not None:
            save_file({'array':array},save_path)
        return array
    
    def match(self,feats0,feats_all:list,thre=0.5):
        scores=[]
        for ft in feats_all: 
            feats1=load_file(ft,device=0)
            scores.append(F.relu(torch.einsum('ld,bcd->blc',feats0.half().cuda(),feats1.half().cuda()).max(dim=-1).values-thre,0).sum(axis=1).cpu())
        return scores