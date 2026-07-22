"""Small shared helpers for the argparse subcommands (argument plumbing only,
no algorithm logic)."""

import json


def resolve_device(device):
    """Turn 'auto' into 'cuda' or 'cpu'; pass through explicit choices."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def add_device_argument(parser, default="auto"):
    parser.add_argument(
        "--device", default=default, choices=["auto", "cuda", "cpu"],
        help="Compute device (default: %(default)s).",
    )


def int_list(value):
    """argparse type= for comma-separated ints, e.g. '16,32,64,128'."""
    return tuple(int(v) for v in value.split(","))


def load_config_overrides(args, config_path):
    """Override an argparse Namespace in-place from a JSON file's keys.

    Keys use the same dashed-to-underscore name as the argparse dest
    (e.g. 'num_iter' for --num-iter). Unknown keys are rejected to catch typos.
    """
    with open(config_path) as fh:
        overrides = json.load(fh)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown option '{key}' in config file '{config_path}'. "
                f"Config keys must match CLI flag names (dashes -> underscores)."
            )
        setattr(args, key, value)
    return args
