from train_test import train_pytorch
from images.dataset import PGMImageDataset
from models.gbrasnet_drop import GBRASNET

if __name__ == '__main__':
    cover_path='../data/BOSSbase-1.01/cover'
    stego_path='../data/BOSSbase-1.01/stego'
    stego_algorithm = 'S-UNIWARD'
    dataset_train = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm)
    dataset_val = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, val=True)
    dataset_test = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, test=True)
    srm_path = '../data/kernels/SRM_Kernels1.npy'
    model = GBRASNET(srm_path=srm_path)
    train_pytorch(model=model,dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, batch_size=16, epochs=100, path_log_base="../data/outputs/torch/drop/S-UNIWARD")
