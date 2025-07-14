import os
import torch
from torch.amp.grad_scaler import GradScaler
import torch.distributed as dist
import yaml
import logging
import time
from datetime import datetime
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import build_dataloader
from losses import build_loss
from models import build_model
from postprocess import build_postprocess
from metrics import build_metric
from optimizers import build_optimizer
from optimizers.lr_scheduler import build_lr_scheduler


# Setup enhanced logging with better formatting
def setup_logger(is_main_process):
    logger = logging.getLogger("Text Recognition Training")
    logger.setLevel(logging.INFO)
    if not logger.handlers and is_main_process:
        handler = logging.StreamHandler()
        # Enhanced formatter with colors and better structure
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluate(
    model,
    val_loader,
    device,
    is_main_process,
    post_process_class,
    eval_class,
    logger,
    scaler,
):
    """
    Enhanced evaluation function that returns metrics for best model tracking.
    This function now returns the evaluation results so we can track the best model.
    """
    if not is_main_process:
        return None

    model.eval()
    logger.info("🔍 Starting evaluation...")

    # Reset metrics for clean evaluation
    eval_class.reset()

    eval_start_time = time.time()
    len(val_loader)

    with torch.no_grad():
        total_frame = 0.0
        sum_images = 0
        with tqdm(
            total=len(val_loader), desc="eval model:", position=0, leave=True
        ) as pbar:
            for _, batch in enumerate(val_loader):
                images = batch[0].to(device)

                # use amp
                if scaler:
                    with torch.autocast(str(device)):
                        output = model(images)
                    output = to_float32(output)
                else:
                    output = model(images)

                batch_numpy = []
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        batch_numpy.append(item.numpy())
                    else:
                        batch_numpy.append(item)

                # Evaluate the results of the current batch
                post_result = None
                post_result = post_process_class(output, batch_numpy[1])
                eval_class(post_result, batch_numpy)

                # Progress indicator for evaluation
                pbar.update(1)
                total_frame += len(images)
                sum_images += 1

        # Get final metrics
        cur_metric = eval_class.get_metric()
        eval_time = time.time() - eval_start_time

        # Format metrics nicely
        metric_strings = []
        for k, v in cur_metric.items():
            if isinstance(v, float):
                metric_strings.append(f"{k}: {v:.4f}")
            else:
                metric_strings.append(f"{k}: {v}")

        logger.info(f"📊 Evaluation Results ({eval_time:.2f}s):")
        logger.info(f"   {' | '.join(metric_strings)}")

    model.train()
    return cur_metric


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    metrics,
    filepath,
    config,
    logger,
    checkpoint_type="latest",
):
    """
    Enhanced checkpoint saving with better logging and error handling.
    """
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict()
            if config["Global"]["distributed"]
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if config["Global"].get("use_amp", False) and scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        filepath = os.path.join(config["Global"]["save_model_dir"], filepath)

        os.makedirs(os.path.dirname(os.path.dirname(filepath)), exist_ok=True)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(checkpoint, filepath)

        # Enhanced logging with emojis and better formatting
        if checkpoint_type == "best":
            logger.info(f"🏆 New best model saved: {filepath}")
            if metrics:
                main_indicator = getattr(metrics, "main_indicator", "accuracy")
                best_value = metrics.get(main_indicator, "N/A")
                logger.info(f"   Best {main_indicator}: {best_value}")
        else:
            logger.info(f"💾 Latest checkpoint saved: {filepath}")

    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint {filepath}: {str(e)}")


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            preds[k] = to_float32(preds[k])
    elif isinstance(preds, list):
        for i in range(len(preds)):
            preds[i] = to_float32(preds[i])
    elif isinstance(preds, torch.Tensor):
        preds = preds.to(dtype=torch.float32)
    return preds


def adaptive_gradient_clipping(model, max_norm=5.0, percentile=95):
    """More intelligent gradient clipping"""
    total_norm = 0
    param_count = 0

    # Calculate gradient norms
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norms.append(param_norm.item())
            total_norm += param_norm.item() ** 2
            param_count += 1

    total_norm = total_norm ** (1.0 / 2)

    # Use percentile-based clipping for more stability
    if grad_norms:
        import numpy as np

        clip_value = min(max_norm, np.percentile(grad_norms, percentile) * 2)
        clip_value = max(clip_value, 0.1)  # Minimum clip value
    else:
        clip_value = max_norm

    # Apply clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

    return total_norm, clip_value


def layer_wise_gradient_clipping(model, base_max_norm=5.0):
    """Different clipping for different layers"""
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    # Clip head (final layer) more aggressively
    if head_params:
        head_norm = torch.nn.utils.clip_grad_norm_(
            head_params, max_norm=base_max_norm * 0.5
        )
        # print(f"Head gradient norm: {head_norm:.4f}")

    # Clip backbone less aggressively
    if backbone_params:
        backbone_norm = torch.nn.utils.clip_grad_norm_(
            backbone_params, max_norm=base_max_norm
        )
        # print(f"Backbone gradient norm: {backbone_norm:.4f}")


def train(config):
    # Distributed training setup
    if config["Global"]["distributed"]:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_str = f"cuda:{local_rank}"
        device = torch.device(device_str)
        is_main_process = dist.get_rank() == 0
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        is_main_process = True

    # Setup logger
    logger = setup_logger(is_main_process)

    if is_main_process:
        logger.info("🚀 Starting Text Recognition Training")
        logger.info(f"   Device: {device_str}")
        logger.info(f"   Distributed: {config['Global']['distributed']}")
        logger.info(f"   AMP: {config['Global'].get('use_amp', False)}")

    criterion = build_loss(config["Loss"]).to(device)

    # Datasets and data loaders
    train_loader = build_dataloader(
        config, mode="Train", logger=logger, seed=config["Global"].get("seed", 17)
    )
    val_loader = build_dataloader(
        config, mode="Eval", logger=logger, seed=config["Global"].get("seed", 17)
    )
    epoch_num = config["Global"]["epoch_num"]

    # AMP setup
    scaler = (
        GradScaler(device=device_str, init_scale=2**10)
        if config["Global"].get("use_amp", False)
        else None
    )

    # Build postprocess and metrics
    post_process_class = build_postprocess(config["PostProcess"], config["Global"])
    char_num = len(getattr(post_process_class, "character"))
    eval_class = build_metric(config["Metric"])

    # Build model, optimizer, and loss
    config["Architecture"]["Backbone"]["vocab_size"] = char_num
    model = build_model(config["Architecture"]["Backbone"]).to(device)
    if config["Global"]["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    optimizer = build_optimizer(config["Optimizer"], model.parameters())
    scheduler = build_lr_scheduler(
        optimizer,
        config["Optimizer"]["lr"],
        epoch_num,
        len(train_loader),
    )

    # Best model tracking - get the main indicator from metrics class
    main_indicator = getattr(eval_class, "main_indicator", "accuracy")
    best_metric_value = float("-inf")  # Assuming higher is better
    best_metrics = None

    if is_main_process:
        logger.info(f"📈 Tracking best model using: {main_indicator}")
        logger.info(
            f"🔄 Training for {epoch_num} epochs with {len(train_loader)} steps per epoch"
        )

    global_step = 0
    start_eval_step, eval_batch_step = config["Global"]["eval_batch_step"]
    print_batch_step = config["Global"]["print_batch_step"]

    # Training loop
    for epoch in range(epoch_num):
        if config["Global"]["distributed"]:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_start_time = time.time()

        if is_main_process:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📚 Epoch {epoch + 1}/{epoch_num}")
            logger.info(f"{'=' * 60}")

        # Training phase
        for step, batch in enumerate(train_loader):
            global_step += 1
            images = batch[0].to(device)

            optimizer.zero_grad()

            if config["Global"].get("use_amp", False):
                with torch.autocast(device_str, dtype=torch.float16):
                    outputs = model(images)
                outputs = to_float32(outputs)
                loss = criterion(outputs, batch)
                avg_loss = loss["loss"]
                scaler.scale(avg_loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                outputs = model(images)

                loss = criterion(outputs, batch)
                avg_loss = loss["loss"]

                # Add loss scaling for stability
                scaled_loss = avg_loss * 0.1  # Scale down loss
                scaled_loss.backward()
                
                # Check for NaN gradients
                nan_count = 0
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in {name}")
                        nan_count += 1
                
                if nan_count > 0:
                    print(f"WARNING: {nan_count} parameters have NaN gradients!")
                    optimizer.zero_grad()
                    return avg_loss.item()
                
                # Apply layer-wise gradient clipping
                layer_wise_gradient_clipping(model, base_max_norm=5.0)

                optimizer.step()
                scheduler.step()

            # Enhanced training logging
            if is_main_process and (step + 1) % print_batch_step == 0:
                batch = [
                    item.numpy() if isinstance(item, torch.Tensor) else item
                    for item in batch
                ]
                post_result = post_process_class(outputs, batch[1])
                metrics = eval_class(post_result, batch)

                # Calculate progress and ETA
                progress = (step + 1) / len(train_loader) * 100
                elapsed = time.time() - epoch_start_time
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)

                # Format training metrics
                metric_str = ""
                for k, v in metrics.items():
                    if isinstance(v, float):
                        metric_str += f" | {k}: {v:.4f}"
                    else:
                        metric_str += f" | {k}: {v}"

                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"⏳ Step {step + 1}/{len(train_loader)} ({progress:.1f}%) | Loss: {avg_loss.item():.4f}{metric_str} | LR: {current_lr:.6f} | ETA: {eta:.0f}s"
                )

        # End of epoch processing
        if (
            is_main_process
            and global_step > start_eval_step
            and (global_step - start_eval_step) % eval_batch_step == 0
        ):
            epoch_time = time.time() - epoch_start_time
            logger.info(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            # Save latest checkpoint
            current_metrics = evaluate(
                model,
                val_loader,
                device,
                is_main_process,
                post_process_class,
                eval_class,
                logger,
                scaler,
            )

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                current_metrics,
                "latest.pth",
                config,
                logger,
                "latest",
            )

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                current_metrics,
                f"iter_epoch_{str(epoch + 1)}.pth",
                config,
                logger,
                "latest",
            )

            # Update best model if needed
            if current_metrics is not None:
                current_value = current_metrics.get(main_indicator, float("-inf"))
                if current_value > best_metric_value:
                    best_metric_value = current_value
                    best_metrics = current_metrics.copy()
                    logger.info(
                        f"🎯 New best {main_indicator}: {best_metric_value:.4f}"
                    )

                    # Save best model checkpoint
                    save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        best_metrics,
                        "best_accuracy.pth",
                        config,
                        logger,
                        "best",
                    )

    # Training completion summary
    if is_main_process:
        logger.info(f"\n{'=' * 60}")
        logger.info("🎉 Training completed successfully!")
        logger.info(f"🏆 Best {main_indicator}: {best_metric_value:.4f}")
        if best_metrics:
            logger.info("📊 Best model metrics:")
            for k, v in best_metrics.items():
                if isinstance(v, float):
                    logger.info(f"   {k}: {v:.4f}")
                else:
                    logger.info(f"   {k}: {v}")
        logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a text recognition model.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "-o",
        nargs="+",
        help="Overwrite configuration values. Format: key1=value1 key2=value2 ...",
    )
    args = parser.parse_args()

    # Load configuration from the specified file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Overwrite configuration values if specified
    if args.o:
        for overwrite in args.o:
            key, value = overwrite.split("=", 1)
            keys = key.split(".")
            sub_config = config
            for k in keys[:-1]:
                sub_config = sub_config.setdefault(k, {})
            # Attempt to parse value as int, float, bool, or list of numbers; otherwise, keep as string
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.startswith("[") and value.endswith("]"):
                        try:
                            value = [
                                int(x) if x.isdigit() else float(x)
                                for x in value[1:-1].split(",")
                            ]
                        except ValueError:
                            pass
            sub_config[keys[-1]] = value

    train(config)
