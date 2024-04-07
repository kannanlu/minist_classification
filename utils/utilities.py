# coding=UTF-8
"""@Description: some useful general functions for neural network 
    @Author: Kannan Lu, lukannan@link.cuhk.edu.hk
    @Date: 2024/04/02
"""
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import torch.utils.data as data


############## device and seed wrappers #################
def set_device(if_mac: bool = False):
    """set the best possible device depending on the system

    Args:
        if_mac (bool, optional): if the system is mac metal gpu. Defaults to False.

    Returns:
        torch.device: the device object.
    """
    if if_mac:
        device = torch.device("mps") if (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()) else torch.device("cpu")
    else:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ", device)
    return device


def check_runtime_performance(device: torch.device,
                              if_mac: bool = False) -> None:
    """compute the run time for cpu, mac gpu and regular gpu using a default task.

    Args:
        device (torch.device): device object. 
        if_mac (bool, optional): if the system is MAC Metal GPU. Defaults to False.
    """
    x = torch.randn(5000, 5000)

    ## CPU version
    start_time = time.time()
    _ = torch.matmul(x, x)
    end_time = time.time()
    print(f"CPU time: {(end_time - start_time):6.5f}s")

    if if_mac:
        ## MAC Metal GPU version
        x = x.to(device)  # assert device is 'mps'
        _ = torch.matmul(x, x)
        # CUDA is asynchronous, so we need to use different timing functions
        start = torch.mps.Event(enable_timing=True)
        end = torch.mps.Event(enable_timing=True)
        start.record()
        _ = torch.matmul(x, x)
        end.record()
        torch.mps.synchronize(
        )  # Waits for everything to finish running on the GPU
        print(f"MAC Metal GPU time: {0.001 * start.elapsed_time(end):6.5f}s"
              )  # Milliseconds to seconds
    else:
        ## GPU version
        x = x.to(device)
        _ = torch.matmul(x, x)  # First operation to 'burn in' GPU
        # CUDA is asynchronous, so we need to use different timing functions
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch.matmul(x, x)
        end.record()
        torch.cuda.synchronize(
        )  # Waits for everything to finish running on the GPU
        print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s"
              )  # Milliseconds to seconds
    return


def set_seed(if_mac: bool = False) -> None:
    """set the manual seed for different systems, cpu, metal gpu and gpu

    Args:
        if_mac (bool, optional): if the system is metal gpu. Defaults to False.
    """
    if if_mac:
        # set metal GPU seed
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(42)

    else:
        # GPU operations have a separate seed we also want to set
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        # Additionally, some operations on a GPU are implemented stochastic for efficiency
        # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Manual seed is set!")
    return


###################### initialization wrappers ################
def kaiming_init(model):
    """initialize model withe Kaiming initialization
    """
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith(
                "layers.0"
        ):  # The first layer does not have ReLU applied on its input
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


def xavier_init(model):
    """initialize model withe Xavier initialization
    """
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


####################### model config, result, state_dict io wrapper######
def _get_config_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


##################### data process wrapper ##############################
class WrappedDataLoader:
    """a wrapper around DataLoader to perform data preprocessing using func.
    """

    def __init__(self, data_loader: data.DataLoader, func: callable) -> None:
        self.data_loader = data_loader
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield (self.func(*batch))
