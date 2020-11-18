import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BaseSequential(nn.Module):
    """
    Sequential neural net with no specified architecture.
    """

    def __init__(self):
        super().__init__()

    def init_weights(self):
        for _layer in self.model:
            if isinstance(_layer, nn.Linear):
                nn.init.kaiming_normal_(_layer.weight, a=0.01)
                nn.init.constant_(_layer.bias, 0.0)

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def eval_per_layer(self, input_tensor):
        """
        Return the input, all intermediates, and the output.
        """
        tensors = [input_tensor]
        current = input_tensor
        self.model.eval()

        for _layer in self.model.children():
            current = _layer(current)
            tensors.append(current)

        return tensors


class MLP(BaseSequential):
    def __init__(self, embed_dim, num_classes, dropout=0.25, n_hid=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_hid),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, num_classes),
        )
        self.init_weights()


class LogisticRegression(BaseSequential):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(embed_dim, num_classes),)
        self.init_weights()
