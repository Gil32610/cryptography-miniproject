from models import GBRASNET
from analysis.plot import Plotter
from images.dataset import PGMImageDataset
import torch
from torch.utils.data import DataLoader
from train_test import pair_collate_fn

if __name__ == '__main__':
    dataset = PGMImageDataset(
    cover_path='../data/BOSSbase-1.01/cover',
    stego_path='../data/BOSSbase-1.01/stego',
    stego_algorithm='WOW',
    bpp='0.4bpp',
    test=True
    )
    plotter = Plotter(dataset=dataset)
    model = GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy')
    trained_model_path = '../data/outputs/torch/WOW_ORGIGINAL/.4bpp/saved-model-034-0.7645.pth'
    state_dict = torch.load(f=trained_model_path)
    model.load_state_dict(state_dict=state_dict)
    models_dict = {
    "GBRASNet WOW 4bpp": model
    }
    test_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=pair_collate_fn, shuffle=False)
    plotter = Plotter(dataset=dataset)
    plotter.compare_integrated_gradients(
        models_dict=models_dict,
        dataloader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_samples=5,
        save_path="../data/outputs/plots/integrated_gradients"
    )