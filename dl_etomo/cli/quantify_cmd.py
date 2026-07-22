"""argparse subcommand: Cliff-Lorimer EDX quantification.
Wraps dl_etomo.quantification.quantify_cl_vol -- no algorithm logic lives here."""

from pathlib import Path

import numpy as np
import tifffile as tiff

from ..quantification import quantify_cl_vol
from ._common import load_config_overrides


def add_parser(subparsers):
    p = subparsers.add_parser(
        "quantify",
        help="Cliff-Lorimer quantification of denormalized EDX intensity maps.",
    )
    p.add_argument("--input-files", required=True,
                    help="Comma-separated paths to denormalized per-element intensity "
                         "TIFFs (e.g. DIP_recon_* outputs of 'dl-etomo dipm-tv'), in the "
                         "same order as --xray-lines.")
    p.add_argument("--xray-lines", required=True,
                    help="Comma-separated X-ray lines matching --input-files 1:1, "
                         "e.g. Ge_Ka,Sb_La,Te_La.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-intensity", type=float, default=0.1)
    p.add_argument("--mask-file", default=None,
                    help="Optional boolean/float mask TIFF applied to every channel.")
    p.add_argument("--auto-mask", action="store_true",
                    help="Compute an Otsu mask on the summed intensity when "
                         "--mask-file is not given.")
    p.add_argument("--config", default=None,
                    help="Optional JSON file overriding any of the flags above.")
    p.set_defaults(func=run)
    return p


def run(args):
    if args.config:
        load_config_overrides(args, args.config)

    input_files = [f.strip() for f in args.input_files.split(",")]
    xray_lines = [x.strip() for x in args.xray_lines.split(",")]
    if len(input_files) != len(xray_lines):
        raise ValueError(
            f"Got {len(input_files)} --input-files but {len(xray_lines)} --xray-lines; "
            f"they must match 1:1 and be given in the same order."
        )

    intensities = [tiff.imread(f).astype(np.float64) for f in input_files]
    mask = tiff.imread(args.mask_file).astype(np.float64) if args.mask_file else None

    atomic_percent = quantify_cl_vol(
        intensities, xray_lines, min_intensity=args.min_intensity,
        mask=mask, auto_mask=args.auto_mask and mask is None,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for xray_line, at_pct in zip(xray_lines, atomic_percent):
        element = xray_line.split("_")[0]
        out_path = out_dir / f"quant_CL_{element}_atpct.tif"
        tiff.imwrite(str(out_path), np.asarray(at_pct, dtype=np.float32))
        print(f"Saved: {out_path}  shape={at_pct.shape}")
