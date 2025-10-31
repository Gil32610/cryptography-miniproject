from models import GBRASNET, GBRASNETStudent
from images.dataset import PGMImageDataset
from analysis.plot import Plotter


if __name__ == '__main__':

    plotter = Plotter(dataset=None)

    configs_dict = {
        "WOW Original 0.2bpp": {
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
        "WOW Student 0.2bpp": {
            "model": (
                GBRASNETStudent(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/Distillation/WOW/.2bpp/saved-model-018-0.6390.pth'
            ),
            "dataset": PGMImageDataset(
                cover_path='../data/BOSSbase-1.01/cover',
                stego_path='../data/BOSSbase-1.01/stego',
                stego_algorithm='WOW',
                bpp='0.2bpp',
                test=True
            )
        },
        "WOW Original 0.4bpp": {
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
        },
        "WOW Student 0.4bpp": {
            "model": (
                GBRASNETStudent(srm_path='../data/kernels/SRM_Kernels1.npy'),
                '../data/outputs/torch/Distillation/WOW/.4bpp/gbras_student_point4bpp_2025-10-30_13-23-20/saved-model-032-0.7515.pth'
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

    plotter.generate_latex_table(
        configs_dict=configs_dict,
        batch_size=4,
        save_path="student metrics_table.tex",
        caption="Performance comparison across steganographic configurations.",
        label="tab:metrics_comparison"
    )
