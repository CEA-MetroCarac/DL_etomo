"""argparse subcommand: classical CS-TV baseline reconstruction.
Wraps dl_etomo.cs_tv -- no algorithm logic lives here.

Requires the optional 'cs-tv' extra (pysap-etomo, modopt); the import is
deferred to run() so that ``dl-etomo --help`` works without it installed.
"""

import numpy as np
import tifffile as tiff

from ._common import add_device_argument, load_config_overrides, resolve_device


def add_parser(subparsers):
    p = subparsers.add_parser(
        "cs-tv",
        help="Classical CS-TV baseline via Condat-Vu primal-dual splitting "
             "(requires: pip install -e '.[cs-tv]').",
    )
    p.add_argument("--input-file", required=True,
                    help="Sinogram TIFF: 2D (n_angles, W) for a single slice, "
                         "or 3D (D, n_angles, W) for the full volume.")
    p.add_argument("--angles-file", required=True, help="Tilt angles file, one value per line.")
    p.add_argument("--output-file", required=True)
    p.add_argument("--lam", type=float, default=None,
                    help="TV regularisation weight. Default: auto-suggested "
                         "from the data via suggest_lambda().")
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--n-power", type=int, default=15)

    add_device_argument(p)
    p.add_argument("--config", default=None,
                    help="Optional JSON file overriding any of the flags above.")
    p.set_defaults(func=run)
    return p


def run(args):
    if args.config:
        load_config_overrides(args, args.config)

    from ..cs_tv import compress_sensing, compress_sensing_2d, suggest_lambda

    device = resolve_device(args.device)
    data = tiff.imread(args.input_file).astype(np.float32)
    angles = np.loadtxt(args.angles_file, dtype=np.float64).reshape(-1)

    lam = args.lam
    if lam is None:
        candidates = suggest_lambda(data)
        lam = candidates[len(candidates) // 2]
        print(f"[cs-tv] auto-suggested lambda={lam:.4g} (candidates: {candidates})")

    if data.ndim == 2:
        reco, costs = compress_sensing_2d(
            data, angles, lam, args.n_iter, n_power=args.n_power, device=device,
        )
    else:
        reco, costs = compress_sensing(
            data, angles, lam, args.n_iter, n_power=args.n_power, device=device,
        )

    tiff.imwrite(args.output_file, np.asarray(reco, dtype=np.float32))
    print(f"Saved: {args.output_file}  shape={reco.shape}  final_cost={costs[-1]:.6g}")
