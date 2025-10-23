from utils import WeightExtractor
from utils import ModelLoader

if __name__ == '__main__':
    output_path = '../outputs/keras/weights/gbrasnet'
    model_path = '../outputs/keras/S-UNIWARD_0.2bpp.hdf5'
    model = ModelLoader.load_tensorflow_model(model_path=model_path)
    WeightExtractor.extract_weights_from_keras(model=model, output_path=output_path)