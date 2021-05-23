import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for l  in range(1, len(sizes)):
            layers += [nn.Linear(sizes[l-1], sizes[l])]
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)