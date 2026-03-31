from layers import Base
from autograd import Grad
from functions import sigmoid

class Sigmoid(Base):
    def forward(self, x):
        return sigmoid(x)

    def parameters(self) -> list[Grad]:
        return []