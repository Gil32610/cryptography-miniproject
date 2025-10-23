from torchvision import transforms 
import torch
import os
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
    
class WeightExtractor:
    @staticmethod
    def extract_weights_from_keras(model:tf.keras.Model, output_path):
        model_name = model.name
        for i, layer in enumerate(model.layers):
            filename = f'{layer.name}_{i}_weights.npy' 
            output_path = os.path.join(output_path, model_name, filename)
            weights = layer.get_weights()
            np.save(file=output_path,arr=weights)
            print(f'Saved {layer.name}')