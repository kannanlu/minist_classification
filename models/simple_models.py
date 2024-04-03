# coding=UTF-8
"""@Description: neural network models
    @Author: Kannan Lu, lukannan@link.cuhk.edu.hk
    @Date: 2024/04/02
"""
import os
import json
import torch.nn as nn
import torch
from utils.utilities import _get_config_file, _get_model_file, _get_result_file
from tqdm.notebook import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter

act_fn_by_name = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
}

loss_fn_by_name = {"cross_entropy": nn.CrossEntropyLoss, "mse": nn.MSELoss}

optim_by_name = {"sgd": optim.SGD}


class SimpleLayer(nn.Module):
    """28*28 to 10 single layer nn
    """

    def __init__(self, act_fn, c_in: int, c_out: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.act_fn = act_fn()
        self.net = nn.Linear(c_in, c_out)
        self.config = {
            "act_fn": self.act_fn.__class__.__name__,
            "input_size": c_in,
            "output_size": c_out,
        }

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        x = self.act_fn(x)
        return x
