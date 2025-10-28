import os 
import torch
import re
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms  
from PIL import Image
from utils import Tensor255


class PGMImageDataset(Dataset):
    
    TRANSFORM = transforms.Compose([
    Tensor255()
    ])
    
    def __init__(self, cover_path, stego_path, stego_algorithm,bpp='0.2bpp', transform=None, train=True, val=False, test=False):
        self.cover_path = cover_path
        self.stego_algorithm = stego_algorithm
        self.stego_path = os.path.join(stego_path, self.stego_algorithm, bpp,'stego')
        self.cover_images = {int(re.findall(r'\d+',f)[0]):f for f in os.listdir(self.cover_path)}
        self.stego_images={int(re.findall(r'\d+',f)[0]):f for f in os.listdir(self.stego_path)}
        self.transform = transform
        if transform is None:
            self.transform = PGMImageDataset.TRANSFORM
        if val or test:
            train = False
        if train:
            self.ix = np.arange(0,4000,1)
        elif val:
            self.ix = np.arange(4000,5000,1)
        elif test:
            self.ix = np.arange(5000,10_000,1)
        
    def __len__(self):
        return len(self.ix)
        
    def __getitem__(self, index):
        index = int(((index - 0) / (len(self.ix)- 1 - 0)) * (self.ix.max() - self.ix.min()) + self.ix.min())
        index +=1
        cover_image = Image.open(os.path.join(self.cover_path,self.cover_images[index])).convert('L')
        stego_image = Image.open(os.path.join(self.stego_path,self.stego_images[index])).convert('L')
        
        if self.transform:
            cover_image = self.transform(cover_image)
            stego_image = self.transform(stego_image)
        
        cover_label = torch.tensor(0,dtype=torch.long)
        stego_label = torch.tensor(1,dtype=torch.long)
        
        return (cover_image, cover_label), (stego_image, stego_label) 
    
    

    

    
        