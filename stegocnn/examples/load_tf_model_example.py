from utils import ModelLoader


if __name__ == '__main__':
    model_path = '../outputs/keras/S-UNIWARD_0.2bpp.hdf5'
    model = ModelLoader.load_tensorflow_model(model_path=model_path)
    print(model)