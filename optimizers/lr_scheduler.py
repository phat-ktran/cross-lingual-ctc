import torch
import math


def build_lr_scheduler(optimizer, config, total_epochs, step_each_epoch):
    """
    Builds a learning rate scheduler with linear warmup + linear cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be attached.
        config (dict): Configuration dictionary containing learning rate settings.
        total_epochs (int): Total number of training epochs.
        step_each_epoch (int): Number of iterations per epoch.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured learning rate scheduler.
    """
    scheduler_type = config["name"]
    total_steps = total_epochs * step_each_epoch
    warmup_epochs = config.get("warmup_epoch", 0)
    warmup_steps = warmup_epochs * step_each_epoch
    base_lr = config["learning_rate"]
    min_lr = base_lr * config.get("min_lr", 1e-5)

    if scheduler_type == "Cosine":

        def linear_warmup_cosine_decay(step):
            if step < warmup_steps:
                # Linear warmup from 0 to base_lr
                return step / warmup_steps
            elif step < total_steps:
                # Cosine decay from base_lr to min_lr
                decay_step = step - warmup_steps
                decay_total = total_steps - warmup_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
                return min_lr / base_lr + (1 - min_lr / base_lr) * cosine_decay
            else:
                return min_lr / base_lr  # After training

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup_cosine_decay
        )
        return scheduler
    else:
        raise NotImplementedError(
            f"Learning rate scheduler {scheduler_type} not supported"
        )
