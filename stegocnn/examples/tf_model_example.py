import numpy as np
import cv2
from utils import ModelLoader

if __name__ == '__main__':
    model_path = '../outputs/keras/S-UNIWARD_0.2bpp.hdf5'
    model = ModelLoader.load_tensorflow_model(model_path=model_path)
    print(model.summary())
    cover = cv2.imread('../data/BOSSbase-1.01/cover/1.pgm', cv2.IMREAD_GRAYSCALE)
    stego = cv2.imread('../data/BOSSbase-1.01/stego/S-UNIWARD/0.2bpp/stego/1.pgm', cv2.IMREAD_GRAYSCALE)
    X = np.array([cover,stego])
    print(model.predict(X))
    