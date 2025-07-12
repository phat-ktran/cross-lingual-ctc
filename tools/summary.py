import os
import sys
import argparse
import yaml
import torch
from torchsummary import summary

# Add the project root directory to the Python path to allow for module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models import build_model
from postprocess import build_postprocess


def display_summary(config, input_size):
    """
    Initializes a model from the given configuration and displays its summary.

    Args:
        config (dict): The configuration dictionary.
        input_size (tuple): The input size for the model, e.g., (C, H, W).
    """
    # To build the model, we need the vocab_size, which is derived from the character set
    # defined in the post-processing configuration. This follows the model setup in train.py.
    post_process_class = build_postprocess(config["PostProcess"], config["Global"])

    # The character attribute can be a string or a list.
    character_set = getattr(post_process_class, "character", "")
    char_num = len(character_set)

    # Update the model architecture configuration with the determined number of characters.
    config["Architecture"]["Backbone"]["vocab_size"] = char_num

    # Build the model using the architecture configuration.
    model = build_model(config["Architecture"]["Backbone"])

    # Determine the device and move the model to it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Displaying model summary for input size {input_size} on device '{device}'")
    print("=" * 60)
    summary(model, input_size, batch_size=3)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a summary of a text recognition model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--input-size",
        "-i",
        nargs=3,
        type=int,
        default=[3, 32, 320],
        help="Input size of the model as C H W. Default: 3 32 320",
    )

    args = parser.parse_args()

    # Load configuration from the specified YAML file.
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error while loading the YAML file: {e}")
        sys.exit(1)

    # Convert the input size from a list to a tuple.
    input_size_tuple = tuple(args.input_size)

    display_summary(config, input_size_tuple)
