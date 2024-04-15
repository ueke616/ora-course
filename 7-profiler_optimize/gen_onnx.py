import onnx
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 1024)
    
    def forward(self, x):
        x = torch.mean(x, 1)
        return self.layer(x)


if __name__ == "__main__":
    model = Model()
    input_ = torch.randn(1024, 1024, 1024)
    output_ = model(input_)
    torch.onnx.export(model, input_, "model.onnx")
