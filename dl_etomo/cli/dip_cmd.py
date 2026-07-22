"""argparse subcommand: 2D single-channel Deep Image Prior reconstruction.
Wraps dl_etomo.dip.dip_reconstruction -- no algorithm logic lives here."""

import numpy as np
import tifffile as tiff
import torch

from ..dip import dip_reconstruction
from ..model import model_unet
from ..radon import sirt_slice_2d
from ._common import add_device_argument, load_config_overrides, resolve_device


def add_parser(subparsers):
    p = subparsers.add_parser(
        "dip-tv", help="2D single-channel Deep Image Prior (+ optional TV) reconstruction.",
    )
    p.add_argument("--input-file", required=True,
                    help="2D sinogram TIFF, shape (n_angles, img_size).")
    p.add_argument("--angles-file", required=True, help="Tilt angles file, one value per line.")
    p.add_argument("--output-file", required=True)

    p.add_argument("--num-iter", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--std-inp-noise", type=float, default=1.0)
    p.add_argument("--noise-reg", type=float, default=0.0)
    p.add_argument("--input-depth", type=int, default=32)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--tv-order", type=int, default=1)
    p.add_argument("--sirt-iter", type=int, default=30,
                    help="SIRT iterations for the visual comparison baseline "
                         "dip_reconstruction expects.")

    add_device_argument(p)
    p.add_argument("--config", default=None,
                    help="Optional JSON file overriding any of the flags above.")
    p.set_defaults(func=run)
    return p


def run(args):
    if args.config:
        load_config_overrides(args, args.config)

    device = resolve_device(args.device)
    sinogram = tiff.imread(args.input_file).astype(np.float32)
    theta = np.loadtxt(args.angles_file, dtype=np.float64).reshape(-1)

    degraded_sirt = sirt_slice_2d(sinogram, theta, n_iter=args.sirt_iter, device=device)

    input_sino = torch.from_numpy(sinogram).float().to(device)[None, None]
    net = model_unet(input_shape=args.input_depth, output_shape=1).to(device)

    result = dip_reconstruction(
        NUM_ITER=args.num_iter, LR=args.lr, IMG_SIZE=args.img_size,
        STD_INP_NOISE=args.std_inp_noise, NOISE_REG=args.noise_reg,
        THETA=theta, INPUT_DEPTH=args.input_depth, net=net,
        input_sino=input_sino, degraded_sirt=degraded_sirt,
        tv_weight=args.tv_weight, tv_order=args.tv_order,
        DISPLAY=False, DEVICE=device,
    )

    tiff.imwrite(args.output_file, np.asarray(result["best_output"], dtype=np.float32))
    print(f"Saved: {args.output_file}  best_loss={result['best_loss']:.6g}  "
          f"best_i={result['best_i']}")
