import os 
import torch
import re
from torch.utils.data import Dataset
from torchvision import transforms  
from PIL import Image


class PGMImageDataset(Dataset):

    def __init__(self, cover_path, stego_path, stego_algorithm,bpp='0.2bpp', transform=None):
        self.cover_path = cover_path
        self.setgo_algorihm = stego_algorithm
        self.cover_images = {re.findall(r'\d+',f)[0]:f for f in os.listdir(self.cover_path)}
        self.stego_images={re.findall(r'\d+',f)[0]:f for f in os.listdir(os.path.join(stego_path,stego_algorithm,bpp,'stego'))}
        self.transform = transform
        
    def __len__(self):
        return len(self.cover_images)
        
    def __getitem__(self, index):
        cover_image = Image.open(self.cover_images[index]).convert('L')
        stego_image = Image.open(self.stego_images[index]).convert('L')
        
        if self.transform:
            cover_image = self.transform(cover_image)
            stego_image = self.transform(stego_image)
        return (cover_image,stego_image)
    
        