import torch
import pandas as pd
from models import GBRASNET, GBRASNETStudent
from images.dataset import PGMImageDataset

# --- Your configurations ---
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


# --- Function to count parameters ---
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --- Collect metrics for LaTeX ---
results = []

for config_name, cfg in configs_dict.items():
    model, model_path = cfg["model"]
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    results.append({
        "Model": config_name,
        "Total Params": f"{total_params:,}",
        "Trainable Params": f"{trainable_params:,}",
    })


# --- Convert to DataFrame ---
df = pd.DataFrame(results)

# --- Convert to LaTeX table ---
latex_table = df.to_latex(index=False, escape=False, caption="Model Parameter Counts", label="tab:model_params")

print(latex_table)
