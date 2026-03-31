from layers import Base

class Model(Base):
    def __init__(self):
        super().__init__()
        self.layers: list[Base] = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
