from analysis.plot import Plotter
from models import GBRASNET
from images.dataset import PGMImageDataset



if __name__ == '__main__':

    plotter = Plotter(dataset=None)  # initialized empty — will switch dynamically

    configs_dict = {
        "S-UNIWARD 0.2bpp": {
            "model": (
                GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/S-UNIWARD_ORIGINAL/.2bpp/saved-model-028-0.6180.pth'
            ),
            "dataset": PGMImageDataset(
                cover_path='../data/BOSSbase-1.01/cover',
                stego_path='../data/BOSSbase-1.01/stego',
                stego_algorithm='S-UNIWARD',
                bpp='0.2bpp',
                test=True
            )
        },
        "WOW 0.2bpp": {
            "model": (
                GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/WOW_ORGIGINAL/.2bpp/saved-model-029-0.6360.pth'
            ),
            "dataset": PGMImageDataset(
                cover_path='../data/BOSSbase-1.01/cover',
                stego_path='../data/BOSSbase-1.01/stego',
                stego_algorithm='WOW',
                bpp='0.2bpp',
                test=True
            )
        },
        "S-UNIWARD 0.4bpp": {
            "model": (
                GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/S-UNIWARD_ORIGINAL/.4bpp/saved-model-036-0.7545.pth'
            ),
            "dataset": PGMImageDataset(
                cover_path='../data/BOSSbase-1.01/cover',
                stego_path='../data/BOSSbase-1.01/stego',
                stego_algorithm='S-UNIWARD',
                bpp='0.4bpp',
                test=True
            )
        },
        "WOW 0.4bpp": {
            "model": (
                GBRASNET(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/WOW_ORGIGINAL/.4bpp/saved-model-034-0.7645.pth'
            ),
            "dataset": PGMImageDataset(
                cover_path='../data/BOSSbase-1.01/cover',
                stego_path='../data/BOSSbase-1.01/stego',
                stego_algorithm='WOW',
                bpp='0.4bpp',
                test=True
            )
        }
    }

    plotter.plot_accuracy_line(configs_dict=configs_dict, batch_size=4, save_path="../data/outputs/plots/accuracy_plot.png")