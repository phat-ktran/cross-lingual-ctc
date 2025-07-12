import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

import copy
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler

from data.imaug import transform, create_operators
from data.simple_dataset import SimpleDataSet, MultiScaleDataSet

# for PaddleX dataset_type
TextRecDataset = SimpleDataSet

__all__ = ["build_dataloader", "transform", "create_operators"]


def build_dataloader(config, mode, logger, seed=None):
    config = copy.deepcopy(config)
    support_dict = ["SimpleDataSet"]

    module_name = config[mode]["dataset"]["name"]

    assert module_name in support_dict, Exception(
        "DataSet only support {}".format(support_dict)
    )
    assert mode in ["Train", "Eval", "Test"], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]
    pin_memory = loader_config.get("pin_memory", True)

    # Datasets and data loaders
    if mode == "Train":
        if dist.is_available() and dist.is_initialized() and config["distributed"]:
            batch_sampler = DistributedSampler(
                dataset=dataset,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            batch_sampler = None
    else:
        batch_sampler = None

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Handle collate function
    collate_fn = None
    if "collate_fn" in loader_config:
        try:
            from . import collate_fn as collate_fn_module

            collate_fn = getattr(collate_fn_module, loader_config["collate_fn"])()
        except ImportError:
            logger.warning(
                "Could not import collate_fn module, using default collate_fn"
            )
            collate_fn = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )