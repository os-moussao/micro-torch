from autograd import Grad
from layers import Base

class ReLU(Base):
    def forward(self, x):
        return Grad.relu(x)
    
    def parameters(self) -> list[Grad]:
        return []