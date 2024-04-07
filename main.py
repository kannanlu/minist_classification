#%%
from utils.data_load import download_dataset
from utils.utilities import (set_device, set_seed, _get_config_file,
                             _get_model_file)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from models.data_format import MNISTDataset
import torch.nn as nn
from torch import optim
from tqdm.notebook import tqdm
from models.simple_models import (SimpleLayer, TwoLayer)
from models.utils import (train_model, train_model_with_logger,
                          train_model_default, save_model, load_model)

# hyperparameters
# activation function
act_fn_by_name = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
}

# loss function
loss_fn_by_name = {"cross_entropy": nn.CrossEntropyLoss, "mse": nn.MSELoss}

# optimizer
optim_by_name = {"sgd": optim.SGD}

# logging directory
logging_dir = "./logs"

# data loader parameters
batch_size = 8
if_shuffle = True
num_workers = 8
if_drop_last = True

# model save/load
# model_name = "TwoLayer"
checkpoint_path = "./checkpoints"

#%% train, get the best result and test
if __name__ == "__main__":
    # set device and seed
    device = set_device(if_mac=False)
    set_seed(if_mac=False)
    # download and prepare MNIST data set
    train_data, valid_data, test_data = download_dataset()

    train_data = MNISTDataset(train_data)
    valid_data = MNISTDataset(valid_data)
    test_data = MNISTDataset(test_data)

    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=if_shuffle,
                                   num_workers=num_workers,
                                   drop_last=if_drop_last)
    valid_loader = data.DataLoader(valid_data,
                                   batch_size=batch_size,
                                   shuffle=if_shuffle,
                                   num_workers=num_workers,
                                   drop_last=if_drop_last)
    test_loader = data.DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=if_shuffle,
                                  num_workers=num_workers,
                                  drop_last=if_drop_last)

    # set up model, loss module, and optimizer
    model = TwoLayer(act_fn_name="sigmoid",
                     c_in=28 * 28,
                     c_hidden=50,
                     c_out=10)
    loss_model = loss_fn_by_name["cross_entropy"]()
    optimizer = optim_by_name["sgd"](model.parameters(), lr=0.01)

    # push the model to device
    model.to(device)

    results = train_model_default(device,
                                  model,
                                  optimizer,
                                  train_loader,
                                  valid_loader,
                                  test_loader,
                                  loss_model,
                                  model_name=model.__class__.__name__,
                                  num_epochs=100,
                                  logging_dir=logging_dir,
                                  checkpoint_path=checkpoint_path,
                                  if_overwrite=True)
