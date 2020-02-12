import json
import logging
import os
import shutil

import pandas as pd
import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_csv(header, data, csv_file):
    """Add a column to 'csv_file'"""
    if os.path.exists(csv_file) is False:
        df = pd.DataFrame(data=data, columns=[header])
    else:
        df = pd.read_csv(csv_file, index_col=0)
        df[header] = data
    df.to_csv(csv_file)


def save_json(params, json_file):
    """Save params dict to a json file"""
    with open(json_file, 'w') as fp:
        json.dump(params, fp, indent=4)


def save_checkpoint(model, model_dir, data_state, optimizer, scheduler):
    """Save model to 'model_dir/pytorch_model.bin', and save data_state to 'model_dir/data_state.pt'

    Args:
        model: (transformers.BertForTokenClassification) model to be saved
        model_dir: (bool) directory for saving model, e.g. 'experiments/conll'
        data_state: (dict) dataset state, in order to restore dataset
        optimizer: (transformers.optimization), in order to restore learning rate
        scheduler: (torch.optim.lr_scheduler), in order to restore learning rate scheduler
    """
    filepath = os.path.join(model_dir, 'model.ckpt')
    state_dict = {
        'data_state': data_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state_dict, filepath)
    model.save_pretrained(model_dir)


def load_checkpoint(model_class, restore_dir, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        model_class: (transformers.BertForTokenClassification) class name
        restore_dir: (string) directory including model
        optimizer: (transformers.optimization) optional: resume optimizer from checkpoint
        scheduler: (torch.optim) optional: resume scheduler from checkpoint
    """
    filepath = os.path.join(restore_dir, 'model.ckpt')
    if not os.path.exists(filepath):
        raise ("File doesn't exist {}".format(filepath))

    state_dict = torch.load(filepath)
    model = model_class.from_pretrained(restore_dir)
    data_state = state_dict['data_state']

    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler:
        scheduler.load_state_dict(state_dict['scheduler'])

    return (model, data_state)
