import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_inputs, n_action, n_hidden_layers=1, hidden_dim=[32]):
        super(MLP, self).__init__()

        M = n_inputs
        self.layers = []
        for hidd_l in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim[hidd_l])
            M = hidden_dim[hidd_l]
            self.layers.append(layer)
            self.layers.append(nn.ReLU())

        # final layer
        self.layers.append(nn.Linear(M, n_action))
        self.layers = nn.Sequential(*self.layers)

        self.losses = None

    def forward(self, X):
        return self.layers(X)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
