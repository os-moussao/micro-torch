from autograd.autograd import Grad
from layers.base import Base

class ReLU(Base):
    def __init__(self):
        pass

    def forward(self, x):
        return Grad.relu(x)
    
    def parameters(self) -> list[Grad]:
        return []