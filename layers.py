import torch
import torch.nn as nn
from typing import List


class Flatten(nn.Module):
    def __init__(self, dim = 1):
        super(Flatten, self).__init__()
        self.dim = dim  # flatten from batch_size dimension

    def forward(self, input):
        return torch.flatten(input, start_dim = self.dim)  # flatten from this dim
        # return input.view(input.size(0), -1)


class RowDynamicKmaxPooling(nn.Module):
    def __init__(self, dim = 1):
        super(Flatten, self).__init__()
        self.dim = dim  # flatten from batch_size dimension

    def forward(self, input):
        return torch.flatten(input, start_dim = self.dim)  # flatten from this dim
        # return input.view(input.size(0), -1)


class Permute(nn.Module):
    def __init__(self, new_view: List[int]):
        super(Permute, self).__init__()
        self.new_view = new_view

    def forward(self, input):
        assert len(input.size()) == len(self.new_view)
        return input.permute(*self.new_view)


class MovingAverage(nn.Module):

    def __init__(self, window_size: int, dimension: int):
        """

        Parameters
        ----------
        window_size: sliding windows size
        dimension: dimension we want to apply sliding window
        """
        super(MovingAverage, self).__init__()
        self.window_size = window_size
        self.dimension = dimension

    def forward(self, input_tensor: torch.Tensor):
        """
        Parameters
        ----------
        input_tensor: torch.Tensor  of shape (B, L, D)
        Returns
        -------
        """
        ret = torch.cumsum(input_tensor, dim = self.dimension)
        ret[:, self.window_size:] = ret[:, self.window_size:] - ret[:, :-self.window_size]
        return ret[:, self.window_size - 1:] / self.window_size
