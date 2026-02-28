from models import GBRASNETStudent, GBRASNET
from analysis.plot import Plotter
from images.dataset import PGMImageDataset
import torch

if __name__ == '__main__':
    dataset = PGMImageDataset(
    cover_path='../data/BOSSbase-1.01/cover',
    stego_path='../data/BOSSbase-1.01/stego',
    stego_algorithm='S-UNIWARD',
    bpp='0.2bpp',
    test=True
    )
    plotter = Plotter(dataset=dataset)
    model = GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy')
    trained_model_path = '../data/outputs/torch/Weight Decay/S-UNIWARD/.2bpp/_2025-10-31_02-24-08/saved-model-031-0.6295.pth'
    state_dict = torch.load(f=trained_model_path)
    print(state_dict.keys())
    model.load_state_dict(state_dict=state_dict)
    metrics = plotter.get_metrics_results(model=model,batch_size=4)
    print(metrics)
    model_name = 'S-UNIWARD 2bpp weight decay model'
    plotter.plot_confusion_matrix(save_path='../data/outputs/plots',model_name=model_name)
    plotter.plot_roc_curve(save_path='../data/outputs/plots',model_name=model_name)
    
