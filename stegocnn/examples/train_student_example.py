from train_test import train_knowledge_distillation
from images.dataset import PGMImageDataset
from models import GBRASNET
from models import GBRASNETStudent
import torch


if __name__ == '__main__':
    cover_path='../data/BOSSbase-1.01/cover'
    stego_path='../data/BOSSbase-1.01/stego'
    stego_algorithm = 'WOW'
    bpp = '0.2bpp'
    dataset_train = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, bpp=bpp)
    dataset_val = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, val=True, bpp=bpp)
    dataset_test = PGMImageDataset(cover_path=cover_path, stego_path=stego_path, stego_algorithm=stego_algorithm, test=True, bpp=bpp)
    srm_path = '../data/kernels/SRM_Kernels1.npy'
    teacher_model = GBRASNET()
    student_model = GBRASNETStudent()
    trained_model_path = '../data/outputs/torch/.2bpp/_2025-10-29_17-26-32/saved-model-043-0.6340.pth'
    state_dict = torch.load(f=trained_model_path)
    teacher_model.load_state_dict(state_dict=state_dict)
    train_knowledge_distillation(teacher=teacher_model, student=student_model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, epochs=100, learning_rate=1e-3, T=2, soft_target_loss_weight=.25, ce_loss_weight=.75, model_name="gbras_student_point2bpp", path_log_base="../data/outputs/torch/student/wow/point2bpp", batch_size=16)
    
    
    
    