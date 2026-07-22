"""argparse subcommand: PSD-based resolution estimation (Lorentzian fit).
Wraps dl_etomo.psd_resolution -- no algorithm logic lives here."""

import json
from pathlib import Path

# compute_psd_analysis's plotting helpers call plt.show() unconditionally,
# which would otherwise try to open a blocking GUI window during a headless
# CLI run. matplotlib.use() must run before pyplot is first imported anywhere
# (including transitively, via the ..psd_resolution import below) to reliably
# switch the backend.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import tifffile as tiff

from ..psd_resolution import compute_psd_analysis, fit_lorentz_cutoff, plot_lorentz_fit
from ._common import load_config_overrides


def add_parser(subparsers):
    p = subparsers.add_parser(
        "psd-resolution",
        help="3D PSD-based resolution estimation via Lorentzian cutoff fitting.",
    )
    p.add_argument("--input-file", required=True, help="Reconstructed volume TIFF (Z, Y, X).")
    p.add_argument("--voxel-nm", type=float, default=0.4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default=None,
                    help="Optional JSON file overriding any of the flags above.")
    p.set_defaults(func=run)
    return p


def _json_default(obj):
    return obj.tolist() if hasattr(obj, "tolist") else str(obj)


def run(args):
    if args.config:
        load_config_overrides(args, args.config)

    vol = tiff.imread(args.input_file).astype(np.float32)
    out = compute_psd_analysis(vol, voxel_nm=args.voxel_nm)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for axis in ("kx", "ky", "kz"):
        profile = out["axis_1d"][axis]
        fit = fit_lorentz_cutoff(profile["k"], profile["smooth"])
        summary[axis] = {k: v for k, v in fit.items() if k != "noise_line"}
        plot_lorentz_fit(out, axis, fit, title=f"{axis} resolution fit",
                          save_path=str(out_dir / f"psd_fit_{axis}.png"))
        if fit.get("ok"):
            print(f"[{axis}] res_cut_nm={fit['res_cut_nm']:.4g}  xi={fit['xi']:.4g}")
        else:
            print(f"[{axis}] fit failed: {fit.get('reason')}")

    with open(out_dir / "psd_resolution.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"Saved PSD resolution fits to {out_dir}")
