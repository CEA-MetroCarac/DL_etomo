import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import scipy
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as _skssim
from scipy.ndimage import rotate as _scipy_rotate

def simplify(x):
    """
    Remove empty dimensions and convert to numpy.

    For a PyTorch tensor: detach from the autograd graph, move to CPU, squeeze,
    and return as a numpy array.
    For a numpy array: squeeze only.
    """
    if torch.is_tensor(x):
        return torch.squeeze(x).detach().cpu().numpy()
    else:
        return np.squeeze(x)

def normalize(x):
    """Normalize a numpy array or torch tensor to [0, 1]."""
    if torch.is_tensor(x):
        return (x - torch.amin(x)) / (torch.amax(x) - torch.amin(x))
    else :
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    
def to8bit(array):
    """Normalize and convert an array to uint8 (0–255)."""
    array = simplify(array)
    array = (normalize(array)*255).astype(np.uint8)
    return array

def sinoToFullView(sinogram, angle):
    """
    Embed a limited-angle sinogram into a 180-row grid, leaving missing angles as zeros.

    Parameters
    ----------
    sinogram : array-like, shape (n_projections, img_size)
        Limited-angle sinogram.
    angle : array-like
        Integer tilt angles in degrees (used as row indices into the 180-row output).

    Returns
    -------
    np.ndarray, shape (180, img_size)
    """

    sinogram = simplify(sinogram)
    size = sinogram.shape[1]
    full = np.zeros((180, size))

    for idx, theta in enumerate(angle):
        full[theta, :] = sinogram[idx, :]

    return full    

def get_torch_grad_op(img_shape, order):
    """
    Build a sparse finite-difference gradient operator as a PyTorch sparse tensor.

    Adapted from pysap-etomo: https://github.com/CEA-COSMIC/pysap-etomo

    Parameters
    ----------
    img_shape : tuple of int
        (H, W) of the square image (H must equal W).
    order : int
        Finite-difference order (1 = standard gradient, 2 = second-order).

    Returns
    -------
    torch.sparse_coo_tensor, shape (2*H*W, H*W)
    """

    img_size = img_shape[0]
    filt = np.zeros((order + 1, 1))
    for k in range(order + 1):
        filt[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

    offsets_x = np.arange(order + 1)
    offsets_y = img_size * np.arange(order + 1)
    shape = (img_size ** 2,) * 2
    sparse_mat_x = scipy.sparse.diags(filt,
                                      offsets=offsets_x, shape=shape).astype(np.float32)
    sparse_mat_y = scipy.sparse.diags(filt,
                                      offsets=offsets_y, shape=shape).astype(np.float32)
    op_matrix = scipy.sparse.vstack([sparse_mat_x, sparse_mat_y])

    indices = torch.from_numpy(np.vstack((op_matrix.row, op_matrix.col)))
    values = torch.from_numpy(op_matrix.data)
    size = torch.Size(op_matrix.shape)
    pytorch_op = torch.sparse_coo_tensor(indices, values, size)

    return pytorch_op

def compute_sparse_tv(img, op):
    """
    Isotropic TV of a 2D image using the sparse gradient operator from ``get_torch_grad_op``.

    Parameters
    ----------
    img : torch.Tensor, shape (H, W)
    op : torch.sparse_coo_tensor
        Operator returned by ``get_torch_grad_op((H, W), order)``.

    Returns
    -------
    torch.Tensor scalar
    """
    grad_torch = torch.sparse.mm(op, img.reshape((img.shape[0]*img.shape[0],1)))
    grad_torch = grad_torch.reshape(2*img.shape[0], img.shape[0])
    
    tv = torch.sum(torch.pow(torch.pow(grad_torch[:img.shape[0],:], 2) + torch.pow(grad_torch[img.shape[0]:,:], 2) + 1e-9, 0.5))
    
    return tv

def _ms_ssim(x, y, data_range=1.0):
    """Multi-scale SSIM in pure PyTorch (Wang et al. 2003)."""
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device)
    ks, sigma = 11, 1.5
    t = torch.arange(ks, dtype=torch.float32, device=x.device) - ks // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g /= g.sum()
    ch = x.shape[1]
    kernel = (g[:, None] * g[None, :])[None, None].expand(ch, 1, ks, ks).contiguous()
    C1, C2, pad = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2, ks // 2

    mcs = []
    for i in range(len(weights)):
        mu_x = F.conv2d(x, kernel, padding=pad, groups=ch)
        mu_y = F.conv2d(y, kernel, padding=pad, groups=ch)
        sig_x  = F.conv2d(x * x, kernel, padding=pad, groups=ch) - mu_x ** 2
        sig_y  = F.conv2d(y * y, kernel, padding=pad, groups=ch) - mu_y ** 2
        sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=ch) - mu_x * mu_y
        cs = (2 * sig_xy + C2) / (sig_x + sig_y + C2)
        ssim_map = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1) * cs
        mcs.append(ssim_map.mean() if i == len(weights) - 1 else cs.mean())
        if i < len(weights) - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)

    return (torch.stack(mcs) ** weights).prod()


def norm_percentile(v, p_lo=1, p_hi=99):
    """Percentile stretch to [0, 1] — works on any numpy array."""
    lo, hi = np.percentile(v, p_lo), np.percentile(v, p_hi)
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def make_rgb_overlay(r, g, b, p_lo=1, p_hi=99):
    """Build an RGB image from three volumes/slices after percentile stretch.

    Pass the same array for two channels to get additive colours, e.g.:
        make_rgb_overlay(fe3, feo, fe3)   → magenta (R+B) / green (G)
        make_rgb_overlay(ge,  sb,  te)    → red / green / blue
    """
    return np.stack([
        norm_percentile(r, p_lo, p_hi),
        norm_percentile(g, p_lo, p_hi),
        norm_percentile(b, p_lo, p_hi),
    ], axis=-1)


def l1_mssim_loss(gt, pred, alpha=0.5):
    """
    Mixed L1 + MS-SSIM loss (H. Zhao et al., "Loss functions for image restoration").

    Parameters
    ----------
    gt, pred : torch.Tensor, shape (1, 1, H, W)
    alpha : float
        Weight of the MS-SSIM term; (1-alpha) weights L1. 0 → pure L1, 1 → pure MS-SSIM.

    Returns
    -------
    torch.Tensor scalar
    """
    msssim_loss = 1 - _ms_ssim(torch.abs(gt), torch.abs(pred), data_range=1.0)
    l1_loss = nn.L1Loss()(gt, pred)

    if alpha < 1e-10: return l1_loss
    if alpha > 1 - 1e-10: return msssim_loss

    return alpha * msssim_loss + (1 - alpha) * l1_loss


# ── Volume / sinogram helpers ─────────────────────────────────────────────

def norm_slice(vol, sl):
    """Slice a (Y,X,Z) or (1,Y,X,Z) volume and normalise to [0,1]."""
    if torch.is_tensor(vol): vol = vol.detach().cpu().numpy()
    if vol.ndim == 4: vol = vol[0]
    x = vol[sl].astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def vol_slice(vol, axis, idx):
    """Extract a 2-D slice from a (Y,X,Z) volume or tensor."""
    if torch.is_tensor(vol): vol = vol.detach().cpu().numpy()
    if vol.ndim == 4: vol = vol[0]
    s = [slice(None)] * 3; s[axis] = idx
    return vol[tuple(s)].astype(np.float32)


def embed_sino(sino_lim, theta_lim, n_angles=180):
    """Embed limited (Y, N_lim, det) sinogram → (Y, n_angles, det) with zeros at missing angles."""
    if torch.is_tensor(sino_lim): sino_lim = sino_lim.cpu().numpy()
    Y, _, W = sino_lim.shape
    out = np.zeros((Y, n_angles, W), dtype=np.float32)
    for i, th in enumerate(theta_lim.astype(int)):
        out[:, th, :] = sino_lim[:, i, :]
    return out


# ── Phantom building helpers ──────────────────────────────────────────────

def paint_sphere(volume, yc, xc, zc, r, val):
    """Paint a sphere of radius r and value val into a (Y,X,Z) volume."""
    Ny, Nx, Nz = volume.shape
    y0, y1 = max(int(yc - r), 0), min(int(yc + r + 1), Ny)
    x0, x1 = max(int(xc - r), 0), min(int(xc + r + 1), Nx)
    z0, z1 = max(int(zc - r), 0), min(int(zc + r + 1), Nz)
    yy = np.arange(y0, y1)[:, None, None]
    xx = np.arange(x0, x1)[None, :, None]
    zz = np.arange(z0, z1)[None, None, :]
    volume[y0:y1, x0:x1, z0:z1][(yy - yc)**2 + (xx - xc)**2 + (zz - zc)**2 <= r**2] = val


def paint_ellipsoid(volume, yc, xc, zc, ry, rx, rz, val=1.0):
    """Paint an ellipsoid with semi-axes (ry, rx, rz) into a (Y,X,Z) volume."""
    Ny, Nx, Nz = volume.shape
    y0, y1 = max(int(np.floor(yc - ry)), 0), min(int(np.ceil(yc + ry)) + 1, Ny)
    x0, x1 = max(int(np.floor(xc - rx)), 0), min(int(np.ceil(xc + rx)) + 1, Nx)
    z0, z1 = max(int(np.floor(zc - rz)), 0), min(int(np.ceil(zc + rz)) + 1, Nz)
    yy, xx, zz = np.ogrid[y0:y1, x0:x1, z0:z1]
    mask = ((yy - yc) / ry)**2 + ((xx - xc) / rx)**2 + ((zz - zc) / rz)**2 <= 1.0
    volume[y0:y1, x0:x1, z0:z1][mask] = val


def rotate_rebinarize(vol, rotations, thr, high):
    """Apply a sequence of (angle, axes) rotations, then threshold to {0, high}."""
    out = vol.copy()
    for ang, ax in rotations:
        out = _scipy_rotate(out, angle=ang, axes=ax, reshape=False, order=1)
    return np.where(out > thr, high, 0.0).astype(np.float32)


# ── Metrics ───────────────────────────────────────────────────────────────

def nrmse(ref, est):
    """Normalised RMSE (range-normalised)."""
    rng = float(np.max(ref) - np.min(ref))
    return float(np.sqrt(np.mean((ref - est) ** 2)) / (rng + 1e-9))


def ssim_score(ref, est):
    """SSIM after percentile stretch to [0,1]."""
    return float(_skssim(norm_percentile(ref), norm_percentile(est), data_range=1.0))


# ── RGB visualisation helpers ─────────────────────────────────────────────

def rgb_channel(v, color, p_lo=1, p_hi=99):
    """Single-channel percentile-stretched RGB image.

    color: 'red' | 'green' | 'blue' | 'magenta'
    """
    x = norm_percentile(v, p_lo, p_hi)
    z = np.zeros_like(x)
    ch = {'red': [x, z, z], 'green': [z, x, z], 'blue': [z, z, x], 'magenta': [x, z, x]}
    return np.stack(ch[color], axis=-1)


def rgb_two_ch(v1, v2, color1, color2, p_lo=1, p_hi=99):
    """Additive mix of two single-channel RGB images (e.g. green + blue)."""
    return np.clip(rgb_channel(v1, color1, p_lo, p_hi) + rgb_channel(v2, color2, p_lo, p_hi), 0, 1)


# ── Comparison figure helpers ─────────────────────────────────────────────

def panel_results(vol_lists, method_names, gt_list, rows, axis, slice_idx,
                  perc=(1, 99), title='', figsize_per_cell=(3, 3), dpi=150):
    """
    Grid comparison figure: columns = methods, rows = channel-group composites.

    Parameters
    ----------
    vol_lists : list of lists
        ``vol_lists[method_idx][channel_idx]`` → 3D volume array.
    method_names : list of str
        Column headers (e.g. ['SIRT', 'CS-TV', 'DIPm-TV']).
    gt_list : list of np.ndarray
        Ground-truth volumes for NRMSE/SSIM computation.
    rows : list of (label, rgb_fn, gt_ch_indices)
        Each row tuple: human-readable label, a callable
        ``rgb_fn(sl_vols, p_hi) → H×W×3`` that composites channel slices, and
        a list of channel indices in ``gt_list`` used for the metric.
    axis : int
        Spatial axis along which to slice (0, 1, or 2).
    slice_idx : int
        Slice index along ``axis``.
    perc : (float, float)
        Percentile stretch applied before ``rgb_fn``.
    title : str
        Figure suptitle.
    figsize_per_cell : (float, float)
        Size in inches of each subplot cell.
    dpi : int
        Figure DPI.
    """
    ncols, nrows = len(method_names), len(rows)
    fw, fh = figsize_per_cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw * ncols, fh * nrows), dpi=dpi)
    if nrows == 1: axes = axes[np.newaxis, :]
    if ncols == 1: axes = axes[:, np.newaxis]

    p_hi = perc[1]
    gt_sl = [vol_slice(g, axis, slice_idx) for g in gt_list]

    for c, (mname, vols) in enumerate(zip(method_names, vol_lists)):
        sl_vols = [vol_slice(v, axis, slice_idx) for v in vols]
        for r, (row_lbl, rgb_fn, gt_chs) in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(rgb_fn(sl_vols, p_hi), interpolation='nearest')
            ax.axis('off')
            if r == 0:
                ax.set_title(mname, fontsize=9, pad=5)
            ax.text(0.02, 0.97, row_lbl, transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color='w',
                    bbox=dict(facecolor='k', alpha=0.4, edgecolor='none', pad=1.5))
            if c > 0:
                nrms  = [nrmse(gt_sl[i], sl_vols[i]) for i in gt_chs]
                ssims = [ssim_score(gt_sl[i], sl_vols[i]) for i in gt_chs]
                ax.text(0.03, 0.05,
                        f'NRMSE {np.mean(nrms):.2f}  SSIM {np.mean(ssims):.2f}',
                        transform=ax.transAxes, fontsize=7, color='w')

    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    if title:
        plt.suptitle(title, fontsize=9, y=1.01)
    plt.show()


def show_channels(gt_list, sirt_ref_list, sirt_lim_list, dipmtv, ch_names, slices, title):
    """Per-channel 4-row comparison: GT | SIRT ref | SIRT limited | DIPm-TV."""
    row_labels = ['GT', 'SIRT ref (180 proj.)', 'SIRT limited', 'DIPm-TV']
    all_data = [gt_list, sirt_ref_list, sirt_lim_list,
                [dipmtv[i] for i in range(dipmtv.shape[0])]]
    for ch_idx, ch_name in enumerate(ch_names):
        fig, axes = plt.subplots(4, len(slices), figsize=(4.5 * len(slices), 16))
        for r, (row_lbl, data_list) in enumerate(zip(row_labels, all_data)):
            for c, (sl, sl_lbl) in enumerate(slices):
                axes[r, c].imshow(norm_slice(data_list[ch_idx], sl))
                axes[r, c].set_title(f'{row_lbl} — {sl_lbl}', fontsize=8)
                axes[r, c].axis('off')
        plt.suptitle(f'{title}\nChannel {ch_idx}: {ch_name}', fontsize=9)
        plt.tight_layout(); plt.show()