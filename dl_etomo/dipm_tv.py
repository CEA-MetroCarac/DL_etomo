"""
dipm_tv.py
==========
Deep Image Prior with multi-channel formulation and Total Variation (DIPm-TV) for 3D EDX and EELS tomographic reconstruction.
 
Contains:
  - 3D U-Net architecture (CNN3D) with configurable encoder/decoder/skip channels
  - TV loss functions
  - Training loop: run_dipm_tv()
  - Data pre-processing helper: preprocess_sinograms()
  - Result saving helper: save_results()
 
How to import:
    from dl_etomo.dipm_tv import CNN3D, run_dipm_tv, preprocess_sinograms, save_results
"""

import numpy as np
import time
import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import tifffile as tiff
import matplotlib.pyplot as plt
from IPython import display, get_ipython
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3D U-Net building blocks
# ---------------------------------------------------------------------------

class DB3D(nn.Module):
    """Encoder block: strided Conv3d + BN + LeakyReLU x2."""
    def __init__(self, in_chan, out_chan, kernel=3, pad_mode='zero'):
        super().__init__()
        self.convblock = nn.Sequential(
            # Strided convolution replaces pooling for downsampling
            nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(),
            nn.Conv3d(out_chan, out_chan, kernel_size=kernel,
                      padding=kernel // 2, padding_mode=pad_mode),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.convblock(x)

class UB3D(nn.Module):
    """Decoder block: Upsample + skip-concat + Conv3d x2."""
    def __init__(self, in_chan, out_chan, skip_chan, kernel=3,
                 up_mode='trilinear', pad_mode='zero'):
        super().__init__()
        self.up_kwargs = (dict(mode=up_mode) if up_mode == 'nearest'
                          else dict(mode=up_mode, align_corners=False))
        self.convblock = nn.Sequential(
            nn.BatchNorm3d(in_chan + skip_chan),
            nn.Conv3d(in_chan + skip_chan, out_chan, kernel_size=kernel,
                      padding=kernel // 2, padding_mode=pad_mode),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(),
            nn.Conv3d(out_chan, out_chan, kernel_size=1, padding_mode=pad_mode),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(),
        )

    def forward(self, x, skip):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], **self.up_kwargs)
            x = torch.cat((x, skip), dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, **self.up_kwargs)
        return self.convblock(x)

class CNN3D(nn.Module):
    """
    3D U-Net for DIP reconstruction of multi-channel EDX tomograms.

    Parameters
    ----------
    nbr : int
        Number of EDX-EELS channels (output channels).
    input_shape : int
        Number of input noise channels.
    down_filters, up_filters, skip_filters : list of int
        Channel widths at each depth level.
    down_kernels, up_kernels, skip_kernels : list of int
        Kernel sizes at each depth level.
    up_mode : str
        Upsampling mode ('trilinear' recommended for 3D).
    pad_mode : str
        Padding mode for convolutions ('reflect' avoids border artefacts).
    force_non_zeros : bool
        If True, square the output to enforce non-negativity.
    """

    def __init__(self,
                 nbr=3,
                 input_shape=32,
                 down_filters=(16, 32, 64, 128),
                 up_filters=(16, 32, 64, 128),
                 skip_filters=(16, 16, 16, 16),
                 down_kernels=(3, 3, 3, 3),
                 up_kernels=(3, 3, 3, 3),
                 skip_kernels=(1, 1, 1, 1),
                 up_mode='trilinear',
                 pad_mode='reflect',
                 force_non_zeros=True):
        super().__init__()

        assert len(down_filters) == len(up_filters) == len(skip_filters) \
               == len(down_kernels) == len(up_kernels) == len(skip_kernels)

        self.depth = len(down_filters)
        self.nbr = nbr
        self.force_non_zeros = force_non_zeros
        self.down_filters = list(down_filters)
        self.up_filters = list(up_filters)
        self.skip_filters = list(skip_filters)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleDict()

        for idx in range(self.depth):
            in_d = input_shape if idx == 0 else down_filters[idx - 1]
            self.down_layers.append(
                DB3D(in_d, down_filters[idx], kernel=down_kernels[idx],
                     pad_mode=pad_mode)
            )
            in_u = (down_filters[-1] if idx == self.depth - 1
                    else up_filters[idx + 1])
            self.up_layers.append(
                UB3D(in_u, up_filters[idx], skip_filters[idx],
                     kernel=up_kernels[idx], up_mode=up_mode,
                     pad_mode=pad_mode)
            )
            if skip_filters[idx] != 0:
                in_s = input_shape if idx == 0 else down_filters[idx - 1]
                self.skip_layers[str(idx)] = nn.Sequential(
                    nn.Conv3d(in_s, skip_filters[idx],
                              kernel_size=skip_kernels[idx],
                              padding=skip_kernels[idx] // 2,
                              padding_mode=pad_mode),
                    nn.BatchNorm3d(skip_filters[idx]),
                    nn.LeakyReLU(),
                )

        # Output head: two conv layers ending with LeakyReLU
        self.edx_conv = nn.Sequential(
            nn.Conv3d(up_filters[0], 16, kernel_size=3,
                      padding=1, padding_mode=pad_mode),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, self.nbr, kernel_size=3,
                      padding=1, padding_mode=pad_mode),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        temp_skip = []

        # --- Encoder ---
        for idx, block in enumerate(self.down_layers):
            temp_skip.append(
                None if self.skip_filters[idx] == 0
                else self.skip_layers[str(idx)](x)
            )
            x = block(x)

        # --- Decoder ---
        for idx, block in enumerate(reversed(self.up_layers)):
            x = block(x, temp_skip[self.depth - idx - 1])

        x = self.edx_conv(x)

        if self.force_non_zeros:
            x = torch.square(x)

        return x

# ---------------------------------------------------------------------------
# TV loss functions
# ---------------------------------------------------------------------------
def compute_tv_3d(x, beta=0.5):
    """
    Isotropic 3D Total Variation loss
    
    Parameters
    ----------
    x : torch.Tensor  shape (1, 1, D, H, W)
    beta : float  exponent (0.5 → sqrt-TV, 1 → standard TV)
    """
    dh = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2)
    dw = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2)
    dz = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2)
    return torch.sum(
        torch.pow(dh[:, :, :-1, :-1, :] + dw[:, :, :-1, :, :-1]
                  + dz[:, :, :, :-1, :-1] + 1e-9, beta)
    )

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def preprocess_sinograms(data, theta, device='cuda'):
    """
    Normalise sinogram data per channel to [0, 1] and convert to a torch tensor.

    Parameters
    ----------
    data : np.ndarray, shape (N_ch, N_angles, H, W) or (N_angles, H, W)
        Raw sinogram data.  Single-channel input is expanded to (1, N_angles, H, W).
        Negative values are clamped to zero.
    theta : list of float
        Tilt angles in degrees (used only to document the angular convention;
        no arithmetic is performed here).
    device : str
        PyTorch device string.

    Returns
    -------
    sino_torch : torch.Tensor, shape (N_ch, H, N_angles, W)
        Normalised sinogram in (channel, depth, angle, detector) order,
        ready to be passed to ``run_dipm_tv``.
    x_min : np.ndarray, shape (N_ch,)
        Per-channel minimum used for normalisation.
    x_max : np.ndarray, shape (N_ch,)
        Per-channel maximum used for normalisation.
    """
    data = data.astype(np.float32)
    if data.ndim == 3:
        data = data[np.newaxis]  # (N_angles, H, W) → (1, N_angles, H, W)
    data[data < 0] = 0.0

    x_min = np.array([data[i].min() for i in range(data.shape[0])])
    x_max = np.array([data[i].max() for i in range(data.shape[0])])

    for i in range(data.shape[0]):
        rng = x_max[i] - x_min[i]
        if rng > 0:
            data[i] = (data[i] - x_min[i]) / rng

    sino_torch = torch.from_numpy(data).float().to(device)
    sino_torch = rearrange(sino_torch, 'n p z x -> n z p x')

    return sino_torch, x_min, x_max

def _norm(x):
    """Normalize a numpy array to [0, 1] for display purposes."""
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_dipm_tv(net, rad_op, sino_torch,
                sirt_vol=None,
                num_iter=1500,
                input_depth=32,
                depth=176,
                img_size=112,
                lr=5e-4,
                noise_reg=0.01,
                exp_weight=0.99,
                lambda_tv=0.0,
                loss_type='L2',
                std_inp_noise=1.0,
                weight_decay=0.01,
                plot_every=25,
                plot_slice=0,
                plot_axis=0,
                save_every=100,
                use_amp=False,
                device='cuda'):
    """
    Run DIP-MTV training loop for multi-channel 3D EDX reconstruction.

    Parameters
    ----------
    net : CNN3D  (already on GPU)
    rad_op : Radon3D  forward/backward projector
    sino_torch : torch.Tensor  shape (N, depth, N_angles, img_size)
    num_iter : int
    input_depth : int  number of noise channels
    depth, img_size : int  volume spatial dimensions
    lr : float  AdamW learning rate
    noise_reg : float  std of noise added to fixed input each iteration
    exp_weight : float  EMA weight for averaged output
    lambda_tv : float  TV regularisation strength (0 = no TV)
    loss_type : 'L1' or 'L2'
    std_inp_noise : float  scale of the fixed uniform input (default 1.0; use 0.1 for EDX)
    weight_decay : float  AdamW weight decay (default 0.0)
    plot_every : int  refresh live plot every N iterations; set to 0 to
        disable (also auto-disabled outside a Jupyter/IPython kernel, e.g.
        when running headlessly from the CLI)
    plot_slice : int  index along plot_axis used in the live preview
    plot_axis : int  volume axis to slice for preview (0=D, 1=H, 2=W)
    save_every : int  store output every N iterations (reduces CPU RAM)
    use_amp : bool  enable automatic mixed precision (reduces GPU VRAM ~50%)
    device : str

    Returns
    -------
    iter_output : list of (iteration, np.ndarray)  sampled reconstructed volumes
                  each array has shape (N_ch, D, H, W)
    loss_values : list of float
    """
    nbr = sino_torch.shape[0]
    device_type = torch.device(device).type

    # --- Loss and optimiser ---
    if loss_type == 'L1':
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # --- Fixed random input noise ---
    net_input_orig = (torch.zeros(1, input_depth, depth, img_size, img_size).uniform_()
                      * std_inp_noise).to(device)

    # --- Network info ---
    n_params = sum(p.numel() for p in net.parameters())
    vol_size = nbr * depth * img_size * img_size
    print(f'Output / Input img size : {net_input_orig.shape}')
    print(f'network parameters: {n_params:_}')
    print(f'image size : {vol_size:_}')

    # --- State ---
    loss_values = []
    iter_output = []
    out_avg = None

    # --- Slice helper: vol shape is (N_ch, D, H, W) ---
    _slice_vol = {
        0: lambda v, s: v[:, s, :, :],    # slice along D
        1: lambda v, s: v[:, :, s, :],    # slice along H
        2: lambda v, s: v[:, :, :, s],    # slice along W
    }[plot_axis]

    # --- Live plot (only inside a Jupyter/IPython kernel; skipped headlessly) ---
    live_plot = plot_every > 0 and get_ipython() is not None
    if live_plot:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        dh = display.display(fig, display_id=True)
        ax[0].set_yscale('log')

    net.train()
    start = time.time()

    pbar = tqdm(range(num_iter), desc='DIP-MTV')
    for it in pbar:
        optimizer.zero_grad(set_to_none=True)

        if noise_reg > 0:
            net_input = net_input_orig + torch.empty_like(net_input_orig).normal_() * noise_reg
        else:
            net_input = net_input_orig

        with torch.amp.autocast(device_type, enabled=use_amp):
            out = net(net_input)  # (1, N_ch, D, H, W)

            # Exponential moving average
            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

            # Forward project each channel → stack avoids intermediate list copy
            temp_sino = torch.stack([rad_op(out[0, idx]) for idx in range(nbr)])

            total_loss = criterion(temp_sino, sino_torch)

            if lambda_tv > 0:
                out_sum = out[0].sum(dim=0)  # (D, H, W)
                total_loss = total_loss + lambda_tv * compute_tv_3d(
                    out_sum[None, None]
                )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = total_loss.item()
        loss_values.append(loss_val)
        pbar.set_postfix(loss=f'{loss_val:.4g}')

        # --- Live preview ---
        if live_plot and (it + 1) % plot_every == 0:
            with torch.no_grad():
                vol = out[0].detach().float().cpu().numpy()

            ax[0].cla()
            ax[0].plot(loss_values)
            ax[0].set_yscale('log')
            ax[0].set_title('loss')

            slices = _slice_vol(vol, plot_slice)  # (N_ch, a, b)
            concat = np.concatenate(
                [slices[i] for i in range(min(nbr, 3))], axis=1
            )
            ax[1].cla()
            ax[1].imshow(np.rot90(concat), cmap='gray', vmin=0)
            ax[1].set_title(f'DIPm-TV it={it+1}  axis={plot_axis} s={plot_slice}')

            ax[2].cla()
            if sirt_vol is not None:
                sirt_np = (sirt_vol.detach().float().cpu().numpy()
                           if torch.is_tensor(sirt_vol)
                           else np.asarray(sirt_vol, dtype=np.float32))
                sirt_slices = _slice_vol(sirt_np, plot_slice)
                concat_sirt = np.concatenate(
                    [_norm(sirt_slices[i]) for i in range(min(nbr, 3))], axis=1
                )
                ax[2].imshow(np.rot90(concat_sirt), cmap='gray', vmin=0.)
                ax[2].set_title('SIRT')

            dh.update(fig)

        # --- Store output ---
        if (it + 1) % save_every == 0 or it == num_iter - 1:
            iter_output.append((it, out[0].detach().float().cpu().numpy()))

        # Free fragmented GPU cache periodically
        if (it + 1) % 100 == 0 and device_type == 'cuda':
            torch.cuda.empty_cache()

    if live_plot:
        plt.close(fig)
    elapsed = time.time() - start
    print(f'Total time: {datetime.timedelta(seconds=elapsed)}')

    return iter_output, loss_values

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_results(iter_output, phase_names, out_dir,
                 iteration=None, x_min=None, x_max=None,
                 sample='', lambda_tv=0.0, lr=5e-4,
                 noise_reg=0.05, loss_type='L2'):
    """
    Save one iteration of the DIP reconstruction as separate TIFF files

    Parameters
    ----------
    iter_output : list of (int, np.ndarray)  returned by run_dipm_tv
    phase_names : list of str  e.g. ['Ge', 'Te', 'Sb']
    out_dir : str or Path
    iteration : int  which iteration to save (default: last); nearest stored is used if exact not found
    x_min, x_max : np.ndarray or None  if provided, denormalises the output
    sample, lambda_tv, lr, noise_reg, loss_type : metadata for filename
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    it_idx, vol = iter_output[-1] if iteration is None else min(
        iter_output, key=lambda x: abs(x[0] - iteration)
    )

    for i, name in enumerate(phase_names):
        ch = np.squeeze(vol[i]).astype(np.float32)   #(1, Z, Y, X) -> (Z, Y, X)

        if ch.ndim != 3:
            raise ValueError(f'{name}: expected a 3D volume, got shape={ch.shape}')

        if x_min is not None and x_max is not None:
            ch = ch * (x_max[i] - x_min[i]) + x_min[i]

        fname = out_dir / (f'DIP_recon_{sample}_{name}_it{it_idx}'
                           f'_tv{lambda_tv}_lr{lr}_nr{noise_reg}_{loss_type}.tif')

        tiff.imwrite(str(fname), ch, imagej=True,
                     metadata={'axes': 'ZYX'})

        print(f'Saved: {fname}  shape(Z,Y,X)={ch.shape}')