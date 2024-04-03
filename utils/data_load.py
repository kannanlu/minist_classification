# coding=UTF-8
"""@Description: some functions for downloading and plotting MNIST dataset
    @Author: Kannan Lu, lukannan@link.cuhk.edu.hk
    @Date: 2024/04/02
"""
import torchvision
from torch.utils.data import random_split
import torch
import numpy as np
import matplotlib.pyplot as plt


def download_dataset():
    """download the MNIST data set in "./data", and split the data set
    """
    train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor())

    train_data, valid_data = random_split(
        train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    print("MNIST data loaded, train size: ", len(train_data),
          "validation size: ", len(valid_data), "test size: ", len(test_data))
    return train_data, valid_data, test_data


def check_dataset(dataset):
    """plot the first 10 data points with label attched from torch.tensor data set
    """
    fig, ax = plt.subplots(10, figsize=(3.375, 20))
    for ii in range(10):
        ax[ii].imshow(np.reshape(dataset[ii][0].numpy(), (28, 28)),
                      cmap="grey")
        ax[ii].set_title(dataset[ii][1])
        ax[ii].axis("off")
    plt.show()
    return
