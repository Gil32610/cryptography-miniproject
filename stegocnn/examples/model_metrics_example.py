from models import GBRASNET
from analysis.plot import Plotter
from images.dataset import PGMImageDataset
import torch

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
    metrics = plotter.get_metrics_results(model=model,batch_size=4)
    print(metrics)
    model_name = 'WOW 4bpp original model'
    plotter.plot_confusion_matrix(save_path='../data/outputs/plots',model_name=model_name)
    plotter.plot_roc_curve(save_path='../data/outputs/plots',model_name=model_name)
    
