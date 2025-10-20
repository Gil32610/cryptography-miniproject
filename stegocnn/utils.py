from torchvision import transforms 
import torch
import numpy as np


class Tensor255:
    
    def __call__(self, img):
        arr = np.array(img,dtype=np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
        