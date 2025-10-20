import os 
import torch
import re
from torch.utils.data import Dataset
from torchvision import transforms  
from PIL import Image



class PGMImageDataset(Dataset):
    TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    def __init__(self, cover_path, stego_path, stego_algorithm,bpp='0.2bpp', transform=None):
        self.cover_path = cover_path
        self.stego_algorithm = stego_algorithm
        self.stego_path = os.path.join(stego_path, self.stego_algorithm, bpp,'stego')
        self.cover_images = {int(re.findall(r'\d+',f)[0]):f for f in os.listdir(self.cover_path)}
        self.stego_images={int(re.findall(r'\d+',f)[0]):f for f in os.listdir(self.stego_path)}
        if transform is None:
            self.transform = PGMImageDataset.TRANSFORM
        
        
        
    def __len__(self):
        return len(self.cover_images)
        
    def __getitem__(self, index):
        index +=1
        cover_image = Image.open(os.path.join(self.cover_path,self.cover_images[index])).convert('L')
        stego_image = Image.open(os.path.join(self.stego_path,self.stego_images[index])).convert('L')
        
        if self.transform:
            cover_image = self.transform(cover_image)
            stego_image = self.transform(stego_image)
        return (cover_image,stego_image)
    
    

    

    
        