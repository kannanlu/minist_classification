# coding=UTF-8
"""@Description: model related train, evaluation, save/load functions
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
from models.simple_models import SimpleLayer
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


def train_model(device,
                model,
                optimizer,
                data_loader,
                loss_module,
                num_epochs=100):
    """train nn model with num_epochs, in each epoch, the data set is divided into different batches 

    Args:
        device (torch.device): the system to operate.
        model (torch.nn.Module): the nn model.
        optimizer (torch.optim): optimizer method.
        data_loader (torch.utils.data.DataLoader): the data loader.
        loss_module (torch.nn.Module): the loss function module.
        num_epochs (int, optional): number of times to perform the training cycle. Defaults to 100.
    """
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            # cast input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            #predict from the model
            preds = model(data_inputs)
            # some dimension setting if necessary
            loss = loss_module(preds, data_labels)

            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()
    return


def train_model_with_logger(device,
                            model,
                            optimizer,
                            data_loader,
                            loss_module,
                            num_epochs=100,
                            logging_dir: str = "./logs"):
    """train nn model with num_epochs, in each epoch, the data set is divided into different batches, the logging is done by tensorboard

    Args:
        device (torch.device): the system to operate.
        model (torch.nn.Module): the nn model.
        optimizer (torch.optim): optimizer method.
        data_loader (torch.utils.data.DataLoader): the data loader.
        loss_module (torch.nn.Module): the loss function module.
        num_epochs (int, optional): number of times to perform the training cycle. Defaults to 100.
        logging_dir (str, optional): logging directory. Defaults to "./logs".
    """
    # create tensorboard logger
    writer = SummaryWriter(log_dir=logging_dir)
    model_graph_documented = False  # flag to store the computational graph for first data point of each epoch
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:

            # cast input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # add the computational graph of the first data point to tensorboard
            if not model_graph_documented:
                writer.add_graph(model, data_inputs)
                model_graph_documented = True

            #predict from the model
            preds = model(data_inputs)
            # some dimension setting if necessary
            loss = loss_module(preds, data_labels)

            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()

            epoch_loss += loss.item()

        # Add average loss to TensorBoard
        epoch_loss /= len(data_loader)
        writer.add_scalar("training_loss", epoch_loss, global_step=epoch + 1)

        # one can also add intermediate figures using writer.add_figure()
    writer.close()
    return


def train_model_default(device,
                        model,
                        optimizer,
                        train_loader,
                        valid_loader,
                        test_loader,
                        loss_module,
                        model_name: str,
                        num_epochs=100,
                        logging_dir: str = "./logs",
                        checkpoint_path: str = "./checkpoint",
                        if_overwrite: bool = False):
    """Perform training over number of epochs, for each epoch, validate the result, and store the best result. 
    the function checks if a previous result exists, if not, start the training; if it exists, depending on if we want
    to overwrite the current result. 

    Args:
        device (torch.device): the system to operate.
        model (torch.nn.Module): the nn model.
        optimizer (torch.optim): optimizer method.
        train_loader (torch.utils.data.DataLoader): the training data loader.
        valid_loader (torch.utils.data.DataLoader): the validation data loader.
        valid_loader (torch.utils.data.DataLoader): the test data loader.
        loss_module (torch.nn.Module): the loss function module.
        model_name (str): the model name for saving and loading.
        num_epochs (int, optional): number of times to perform the training cycle. Defaults to 100.
        logging_dir (str, optional): logging directory. Defaults to "./logs".
        checkpoint_path (str, optional): check point path for store model, result and config. Defaults to "./checkpoint".
        if_overwrite (bool, optional): flag for overwrite the saved model or not. Defaults to False.
    """
    file_exist = os.path.isfile(_get_model_file(checkpoint_path, model_name))
    if file_exist and not if_overwrite:
        print(
            f"Model file of \"{model_name}\" already exists. Skipping training..."
        )
        with open(_get_result_file(checkpoint_path, model_name), "r") as f:
            results = json.load(f)
    else:
        if file_exist:
            print("Model file exists, but will be overwritten...")

        # create tensorboard logger
        writer = SummaryWriter(log_dir=logging_dir)
        model_graph_documented = False  # flag to store the computational graph for first data point of each epoch
        # Set model to train mode
        results = None
        model.train()
        train_scores = []
        valid_scores = []
        best_val_epoch = -1  # store the epoch number showing the best validation result
        for epoch in tqdm(range(num_epochs)):
            # train section
            train_loss = 0.0
            true_pred, count = 0.0, 0.0
            t = tqdm(train_loader, leave=False)
            for data_inputs, data_labels in t:

                # cast input data to device (only strictly necessary if we use GPU)
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                # add the computational graph of the first data point to tensorboard
                if not model_graph_documented:
                    writer.add_graph(model, data_inputs)
                    model_graph_documented = True

                #predict from the model
                preds = model(data_inputs)
                # some dimension setting if necessary
                loss = loss_module(preds, data_labels)

                # Before calculating the gradients, we need to ensure that they are all zero.
                # The gradients would not be overwritten, but actually added to the existing ones.
                optimizer.zero_grad()
                # Perform backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()

                # calculate statistics
                train_loss += loss.item()
                true_pred += (preds.argmax(1) == data_labels).type(
                    torch.float).sum().item()
                count += data_labels.shape[0]

            train_acc = true_pred / count
            train_scores.append(train_acc)
            # Add average loss to TensorBoard
            train_loss /= len(train_loader)
            writer.add_scalar("training_loss",
                              train_loss,
                              global_step=epoch + 1)

            # validation section
            valid_loss, valid_acc = eval_model(device, model, valid_loader,
                                               loss_module)
            valid_scores.append(valid_acc)
            writer.add_scalar("validation_loss",
                              valid_loss,
                              global_step=epoch + 1)

            print(
                f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {valid_acc*100.0:05.2f}%"
            )

            if (len(valid_scores) == 0) or (valid_acc
                                            > train_scores[best_val_epoch]):
                print("\t   (New best performance, saving model...)")
                save_model(model, checkpoint_path, model_name)
                best_val_epoch = epoch

    # if the training is done, calculate the performace on the test data set
    if not results:
        load_model(checkpoint_path, model_name, model)
        test_loss, test_acc = eval_model(device, model, test_loader,
                                         loss_module)
        results = {
            "test_acc": test_acc,
            "val_scores": valid_scores,
            "train_losses": train_loss,
            "train_scores": train_scores
        }
        with open(_get_result_file(checkpoint_path, model_name), "w") as f:
            json.dump(results, f)

    return results


@torch.no_grad()
def eval_model(device, model, data_loader, loss_module):
    """calculate the model loss and accuracy.

    Args:
        device (torch.device): the system to operate.
        model (torch.nn.Module): the nn model.
        data_loader (torch.utils.data.DataLoader): the data loader.
        loss_module (torch.nn.Module): the loss function module.
    Returns:
        (float, float): loss and accuracy
    """
    model.eval()
    size = len(data_loader.dataset)  # number of data points
    num_batches = len(data_loader)
    loss = 0
    true_pred = 0
    count = 0
    # loop over all batches of an epoch
    for data_inputs, data_labels in data_loader:
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)

        preds = model(data_inputs)
        loss += loss_module(
            preds, data_labels).item()  # item select only the loss value
        true_pred += (preds.argmax(1) == data_labels).type(
            torch.float).sum().item()
        count += data_labels.shape[0]

    loss /= num_batches
    acc = true_pred / count
    print("model loss: ", loss, "model accuracy: ", acc)
    return loss, acc


def load_model(model_path, model_name, net=None):
    """load model file and model config 
    """
    config_file, model_file = _get_config_file(model_path,
                                               model_name), _get_model_file(
                                                   model_path, model_name)
    assert os.path.isfile(
        config_file
    ), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(
        model_file
    ), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict["act_fn_name"].lower()
        assert act_fn_name in act_fn_by_name, f"Unknown activation function \"{act_fn_name}\". Please add it to the \"act_fn_by_name\" dict."
        net = SimpleLayer(**config_dict)
    net.load_state_dict(torch.load(model_file))
    return net


def save_model(model, model_path, model_name):
    """save model file and model config
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path,
                                               model_name), _get_model_file(
                                                   model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)
    return
