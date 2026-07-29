"""
dl_etomo.cli
============
Argparse subcommands, one module per reconstruction/analysis entry point.
Each module owns an ``add_parser(subparsers)`` function that registers its
subcommand and a ``run(args)`` function that calls the corresponding
``dl_etomo.*`` function -- this package only does argument plumbing.
"""

import argparse

from . import cs_tv_cmd, dip_cmd, dipm_tv_cmd, psd_cmd, quantify_cmd

_SUBCOMMANDS = (dipm_tv_cmd, dip_cmd, cs_tv_cmd, quantify_cmd, psd_cmd)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dl-etomo",
        description="Deep learning strategies for electron tomography "
                    "(DIP, DIPm-TV, CS-TV, EDX quantification, PSD resolution).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in _SUBCOMMANDS:
        module.add_parser(subparsers)
    return parser
