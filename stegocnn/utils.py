from torchvision import transforms 
import torch
import numpy as np
import tensorflow as tf

class Tensor255:
    def __call__(self, img):
        arr = np.array(img,dtype=np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
    
class ModelLoader:        
    @staticmethod
    def load_tensorflow_model(model_path) -> tf.keras.Model:
        return tf.keras.models.load_model(filepath=model_path, compile=False, custom_objects={'Tanh3': lambda x: 3 * tf.keras.activations.tanh(x)})
        