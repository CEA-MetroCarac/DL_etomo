"""
dip.py
======
2D Deep Image Prior (DIP) reconstruction for tomography.

Adapted from the original DIP work (D. Ulyanov et al.):
    https://github.com/DmitryUlyanov/deep-image-prior

Public API
----------
dip_reconstruction(...)
    Run the DIP training loop on a 2D sinogram.

plot_function(dh, data)
    Live Jupyter display update for training progress.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display

import torch
from torch import nn
from einops import rearrange

from model import model_unet  # noqa: F401 — re-exported for notebook convenience
from radon import Radon2D
from utils import simplify, sinoToFullView, get_torch_grad_op, compute_sparse_tv


def dip_reconstruction(
    NUM_ITER,
    LR,
    IMG_SIZE,
    STD_INP_NOISE,
    NOISE_REG,
    THETA,
    INPUT_DEPTH,
    net,
    input_sino,
    degraded_sirt,
    reference_reco=None,
    tv_weight=0.0,
    tv_order=1,
    SHOW_EVERY=100,
    given_input=None,
    state=None,
    DISPLAY=False,
    DEVICE='cuda',
):
    """
    2D DIP tomographic reconstruction from a sinogram.

    Parameters
    ----------
    NUM_ITER : int
        Number of DIP iterations.
    LR : float
        AdamW learning rate.
    IMG_SIZE : int
        Output image size (square).
    STD_INP_NOISE : float
        Standard deviation of the uniform input noise initialisation.
    NOISE_REG : float
        If > 0, std of per-iteration perturbation added to the fixed input
        (acts as regularisation — analogous to early stopping).
    THETA : array-like
        Tilt angles in degrees corresponding to ``input_sino``.
    INPUT_DEPTH : int
        Number of channels in the input noise tensor.
    net : nn.Module
        Untrained U-Net (``model_unet``).
    input_sino : torch.Tensor, shape (1, 1, n_angles, img_size)
        Target sinogram used to compute the data-fidelity loss.
    degraded_sirt : np.ndarray
        SIRT reconstruction used only for visual comparison during training.
    reference_reco : np.ndarray or None
        Ground-truth or reference image shown alongside the live preview.
    tv_weight : float
        Weight of the optional TV regularisation term (0 = disabled).
    tv_order : int
        Order of the TV gradient operator (1 = standard TV).
    SHOW_EVERY : int
        Live display refresh period (iterations).  Only used when ``DISPLAY=True``.
    given_input : torch.Tensor or None
        Custom fixed input noise.  If None, uniform noise is generated.
    state : dict or None
        Checkpoint dict with keys ``model_state_dict`` and
        ``optimizer_state_dict`` to resume training.
    DISPLAY : bool
        If True, show a live Jupyter figure every ``SHOW_EVERY`` iterations.
    DEVICE : str
        PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    dict
        best_loss       : float — lowest sinogram loss seen during training
        best_output     : np.ndarray — reconstruction at ``best_i``
        best_i          : int — iteration index of the best output
        loss_values     : list[float] — per-iteration loss
        net             : nn.Module — trained network
        out_avg         : np.ndarray — EMA-averaged output (exp_weight=0.99)
        best_input      : np.ndarray — input that produced ``best_output``
        list_iter_reco  : list[np.ndarray] — per-iteration reconstructions
        training_state  : dict — model & optimizer state for resuming
    """
    # --- Input noise ---
    if given_input is None:
        net_input = (torch.zeros([1, INPUT_DEPTH, IMG_SIZE, IMG_SIZE])
                     .uniform_() * STD_INP_NOISE).to(DEVICE)
    else:
        net_input = given_input.clone().to(DEVICE)
    net_input_orig = net_input.clone()

    # --- Network and optimiser ---
    net = net.to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)

    if state is not None:
        net.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

    criterion = nn.MSELoss(reduction='sum').to(DEVICE)
    radon_op = Radon2D(size=IMG_SIZE, angle=np.deg2rad(np.flip(THETA)), device=DEVICE)
    radon_op_full = Radon2D(size=IMG_SIZE, angle=np.deg2rad(np.arange(0., 180., 1.)), device=DEVICE)
    grad_op = get_torch_grad_op((IMG_SIZE, IMG_SIZE), tv_order).to(DEVICE)

    # --- State ---
    loss_values = []
    best_loss = 1e9
    best_output = None
    best_input = None
    best_i = 0
    out_avg = None
    exp_weight = 0.99
    list_iter_reco = []
    dh = None

    for it in tqdm(range(NUM_ITER)):
        optimizer.zero_grad(set_to_none=True)

        if NOISE_REG > 0:
            net_input = net_input_orig + torch.empty_like(net_input_orig).normal_() * NOISE_REG

        out = net(net_input)
        out = rearrange(out, '1 1 h w -> h w')

        out_sino = rearrange(radon_op.forward(out), 'h w -> 1 1 h w')
        total_loss = criterion(out_sino, input_sino)

        if tv_weight > 0:
            total_loss = total_loss + tv_weight * compute_sparse_tv(out, grad_op)

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        loss_values.append(loss_val)
        list_iter_reco.append(simplify(out))

        # Exponential moving average of the output
        with torch.no_grad():
            out_d = out.detach()
            out_avg = out_d if out_avg is None else out_avg * exp_weight + out_d * (1 - exp_weight)

        if loss_val < best_loss:
            best_loss = loss_val
            best_i = it
            best_output = simplify(net(net_input))
            best_input = simplify(net_input)

        if DISPLAY and ((it + 1) % SHOW_EVERY == 0 or it == 0):
            dh = plot_function(dh, [
                loss_values,
                simplify(radon_op_full.forward(out)),
                sinoToFullView(input_sino, np.flip(THETA)),
                simplify(out),
                simplify(degraded_sirt),
                reference_reco,
            ])

    return {
        'best_loss': best_loss,
        'best_output': best_output,
        'best_i': best_i,
        'loss_values': loss_values,
        'net': net,
        'out_avg': simplify(out_avg),
        'best_input': best_input,
        'list_iter_reco': list_iter_reco,
        'training_state': {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
    }


def plot_function(dh, data):
    """
    Live Jupyter display: loss curve, sinograms, and reconstructions.

    Parameters
    ----------
    dh : IPython DisplayHandle or None
        Handle returned by a previous call; None on the first call.
    data : list
        [loss_values, gen_sino, ref_sino, gen_reco, sirt_reco, gt_reco]

    Returns
    -------
    IPython DisplayHandle
        Pass back to the next call so the figure updates in place.
    """
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))

    ax[0][0].plot(data[0], color='blue')
    ax[0][0].set_yscale('log')
    ax[0][0].set_title('Training loss')

    extent = [0, data[3].shape[1], 90, -90]
    ax[0][1].imshow(data[1], cmap='gray', aspect='auto', extent=extent)
    ax[0][1].set_title('Generated DIP sinogram')
    ax[0][2].imshow(data[2], cmap='gray', aspect='auto', extent=extent)
    ax[0][2].set_title('Target sinogram (loss reference)')

    if data[5] is None:
        ax[1][0].set_visible(False)
    else:
        ax[1][0].imshow(simplify(data[5]), cmap='gray')
        ax[1][0].set_title('Reference reconstruction')
        ax[1][0].axis('off')

    ax[1][1].imshow(data[3], cmap='gray')
    ax[1][1].set_title('DIP reconstruction')
    ax[1][1].axis('off')
    ax[1][2].imshow(data[4], cmap='gray')
    ax[1][2].set_title('Degraded SIRT reconstruction')
    ax[1][2].axis('off')

    if dh is None:
        dh = display.display(fig, display_id=True)
    else:
        display.update_display(fig, display_id=dh.display_id)

    plt.close(fig)
    return dh
