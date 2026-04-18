from torch.nn import Module, Dropout2d
from torchvision.ops import DropBlock2d
from abc import ABC, abstractmethod

class DropoutStrategy(ABC):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def __call__(self, x):
        pass

class NoDropout(DropoutStrategy):

    def __call__(self, x):
        return x

class SpatialDropout(DropoutStrategy, Module):

    def __init__(self, **kwargs):
        DropoutStrategy.__init__(**kwargs)
        Module.__init__()
        self.dropout = Dropout2d(self.prob)

    def __call__(self, x):
        return self.dropout(x)
    
class Dropblock(DropoutStrategy, Module):

    def __init__(self, **kwargs):
        DropoutStrategy.__init__(**kwargs)
        Module.__init__()
        self.dropout = DropBlock2d(self.prob, self.block_size)

    def __call__(self, x):
        return self.dropout.forward(x)
