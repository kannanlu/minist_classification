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

    def __init__(self, act_fn_name: str, c_in: int, c_out: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.act_fn = act_fn_by_name[act_fn_name.lower()]()  #pass in low str
        self.net = nn.Linear(c_in, c_out)
        self.config = {
            "act_fn_name":
            self.act_fn.__class__.__name__,  # store names in nn 
            "c_in": c_in,
            "c_out": c_out,
        }

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        x = self.act_fn(x)
        return x


class TwoLayer(nn.Module):
    """28*28 to 50 hiden to 10 MLP nn
    """

    def __init__(self, act_fn_name: str, c_in: int, c_hidden: int,
                 c_out: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.act_fn = act_fn_by_name[act_fn_name.lower()]()  #pass in low str
        self.ly1 = nn.Linear(c_in, c_hidden)
        self.ly2 = nn.Linear(c_hidden, c_out)
        self.config = {
            "act_fn_name":
            self.act_fn.__class__.__name__,  # store names in nn 
            "c_in": c_in,
            "c_hidden": c_hidden,
            "c_out": c_out,
        }

    def forward(self, x):
        x = self.flatten(x)
        x = self.ly1(x)
        x = self.act_fn(x)
        x = self.ly2(x)
        x = self.act_fn(x)
        return x
