import torch.nn as nn
import torch.nn.functional as F

class TanH3(nn.Module):
    def forward(x):
        return 3 * F.tanh(x)