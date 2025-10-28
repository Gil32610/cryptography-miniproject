from images import PGMImageDataset

if __name__ == '__main__':
    cover_path='../data/BOSSbase-1.01/cover'
    stego_path='../data/BOSSbase-1.01/stego'
    stego_algorithm = 'HILL'
    dataset = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm)
    images, labels = dataset[0] 
    print(images.shape, labels.shape)
    print(len(dataset))