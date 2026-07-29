"""argparse subcommand: 3D multi-channel DIPm-TV reconstruction (the flagship
method). Wraps dl_etomo.dipm_tv -- no algorithm logic lives here."""

import numpy as np

from ..dataio import load_tilt_series
from ..dipm_tv import CNN3D, preprocess_sinograms, run_dipm_tv, save_results
from ..radon import Radon3D
from ._common import add_device_argument, int_list, load_config_overrides, resolve_device


def add_parser(subparsers):
    p = subparsers.add_parser(
        "dipm-tv",
        help="3D multi-channel Deep Image Prior + Total Variation reconstruction (DIPm-TV).",
    )

    # --- data ---
    p.add_argument("--input-dir", required=True,
                    help="Directory with per-channel projection stacks (native layout), "
                         "or a dataset root containing derived/ + metadata/ (hf layout).")
    p.add_argument("--dataset-format", default="auto", choices=["auto", "native", "hf"])
    p.add_argument("--prefix", default=None,
                    help="Sample prefix to filter files, e.g. 'SET' or 'Virgin' "
                         "(native, when a folder mixes several samples).")
    p.add_argument("--angles-file", default=None,
                    help="Tilt angles file, one value per line. Required for "
                         "--dataset-format native; ignored for hf.")
    p.add_argument("--phase-names", default=None,
                    help="Comma-separated channel names/order, e.g. Ge,Sb,Te. "
                         "Default: every discovered channel, sorted.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-name", default="",
                    help="Sample tag included in saved TIFF filenames.")

    # --- run_dipm_tv hyperparameters (mirrors its keyword defaults 1:1) ---
    p.add_argument("--num-iter", type=int, default=1500)
    p.add_argument("--input-depth", type=int, default=32)
    p.add_argument("--depth", type=int, default=176)
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--noise-reg", type=float, default=0.01)
    p.add_argument("--exp-weight", type=float, default=0.99)
    p.add_argument("--lambda-tv", type=float, default=0.0)
    p.add_argument("--loss-type", default="L2", choices=["L1", "L2"])
    p.add_argument("--std-inp-noise", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--plot-every", type=int, default=0,
                    help="Live preview refresh period; 0 disables it (default -- this "
                         "is a headless CLI). Also auto-disabled outside a notebook.")
    p.add_argument("--plot-slice", type=int, default=0)
    p.add_argument("--plot-axis", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--use-amp", action="store_true")

    # --- CNN3D architecture ---
    p.add_argument("--down-filters", type=int_list, default=(16, 32, 64, 128))
    p.add_argument("--up-filters", type=int_list, default=(16, 32, 64, 128))
    p.add_argument("--skip-filters", type=int_list, default=(16, 16, 16, 16))
    p.add_argument("--down-kernels", type=int_list, default=(3, 3, 3, 3))
    p.add_argument("--up-kernels", type=int_list, default=(3, 3, 3, 3))
    p.add_argument("--skip-kernels", type=int_list, default=(1, 1, 1, 1))
    p.add_argument("--up-mode", default="trilinear")
    p.add_argument("--pad-mode", default="reflect")

    add_device_argument(p)
    p.add_argument("--config", default=None,
                    help="Optional JSON file overriding any of the flags above "
                         "(keys = flag names with dashes replaced by underscores).")
    p.set_defaults(func=run)
    return p


def run(args):
    if args.config:
        load_config_overrides(args, args.config)

    requested_phases = args.phase_names.split(",") if args.phase_names else None
    sinograms, angles_deg, phase_names = load_tilt_series(
        args.input_dir, format=args.dataset_format, prefix=args.prefix,
        angles_file=args.angles_file, phase_names=requested_phases,
    )

    device = resolve_device(args.device)
    sino_torch, x_min, x_max = preprocess_sinograms(sinograms, angles_deg, device=device)

    rad_op = Radon3D(depth=args.depth, size=args.img_size,
                      angle=np.deg2rad(angles_deg), device=device)

    net = CNN3D(
        nbr=len(phase_names), input_shape=args.input_depth,
        down_filters=args.down_filters, up_filters=args.up_filters,
        skip_filters=args.skip_filters, down_kernels=args.down_kernels,
        up_kernels=args.up_kernels, skip_kernels=args.skip_kernels,
        up_mode=args.up_mode, pad_mode=args.pad_mode,
    ).to(device)

    iter_output, loss_values = run_dipm_tv(
        net, rad_op, sino_torch,
        num_iter=args.num_iter, input_depth=args.input_depth, depth=args.depth,
        img_size=args.img_size, lr=args.lr, noise_reg=args.noise_reg,
        exp_weight=args.exp_weight, lambda_tv=args.lambda_tv,
        loss_type=args.loss_type, std_inp_noise=args.std_inp_noise,
        weight_decay=args.weight_decay, plot_every=args.plot_every,
        plot_slice=args.plot_slice, plot_axis=args.plot_axis,
        save_every=args.save_every, use_amp=args.use_amp, device=device,
    )

    save_results(
        iter_output, phase_names, args.output_dir,
        x_min=x_min, x_max=x_max, sample=args.sample_name,
        lambda_tv=args.lambda_tv, lr=args.lr, noise_reg=args.noise_reg,
        loss_type=args.loss_type,
    )
    print(f"Final loss: {loss_values[-1]:.6g}  ({len(loss_values)} iterations)")
