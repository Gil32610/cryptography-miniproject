from stegocnn import train_test
from images.dataset import PGMImageDataset

if __name__ == '__main__':
    cover_path='../data/BOSSbase-1.01/cover'
    stego_path='../data/BOSSbase-1.01/stego'
    stego_algorithm = 'HILL'
    train_dataset = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm)
    val_dataset = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, val=True)
    test_dataset = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, test=True)