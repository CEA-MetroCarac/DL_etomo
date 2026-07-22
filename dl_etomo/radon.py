"""
radon.py
========
Tomosipo-based Radon transform operators and SIRT reconstruction for 2D and 3D
parallel-beam tomography.

SIRT implementation adapted from the Tomosipo official examples
(A. Hendriksen et al.): https://github.com/ahendriksen/tomosipo

Public API
----------
Radon2D          : 2D forward/backward Radon operator
Radon3D          : 3D forward/backward Radon operator
sirt_slice_2d    : convenience function — SIRT on a single 2D sinogram (numpy in/out)
"""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

import tomosipo as ts
from tomosipo.torch_support import to_autograd


class Radon2D(nn.Module):
    """
    2D parallel-beam Radon transform with SIRT reconstruction.

    Parameters
    ----------
    size : int
        Image size (square: size × size pixels).
    angle : int or array-like
        If int N: N uniformly spaced angles in [0, π).
        If array: exact angles in radians.
    device : str
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(self, size=256, angle=180, device='cuda'):
        super().__init__()
        self.img_size = size
        self.angle = angle
        self.device = device
        self._init_op()

    def _init_op(self):
        self.vg = ts.volume(size=1, shape=(1, self.img_size, self.img_size))
        self.pg = ts.parallel(angles=self.angle, shape=(1, self.img_size), size=(1, 1))
        self.operator = ts.operator(self.vg, self.pg)

    def forward(self, img):
        """
        Forward Radon transform (sinogram generation).

        Parameters
        ----------
        img : torch.Tensor, shape (img_size, img_size)

        Returns
        -------
        torch.Tensor, shape (n_angles, img_size)
        """
        return to_autograd(self.operator, is_2d=True)(img)

    def bp(self, sino):
        """
        Unfiltered backprojection.

        Parameters
        ----------
        sino : torch.Tensor, shape (n_angles, img_size)

        Returns
        -------
        torch.Tensor, shape (img_size, img_size)
        """
        return to_autograd(self.operator.T, is_2d=True)(sino)

    def backward_sirt_ts(self, sino, progress_bar=False, min_constraint=True, num_iters=100):
        """
        SIRT reconstruction from a 2D sinogram.

        Parameters
        ----------
        sino : torch.Tensor, shape (n_angles, img_size)
        progress_bar : bool
            Show tqdm progress bar.
        min_constraint : bool
            Enforce non-negativity at each iteration.
        num_iters : int
            Number of SIRT iterations.

        Returns
        -------
        torch.Tensor, shape (img_size, img_size)
        """
        y = sino.to(self.device)
        x_cur = torch.zeros(self.operator.domain_shape, device=self.device, requires_grad=True)

        C = to_autograd(self.operator.T)(
            torch.ones(self.operator.range_shape, device=self.device, requires_grad=True))
        C[C < ts.epsilon] = np.inf
        C.reciprocal_()

        R = to_autograd(self.operator)(
            torch.ones(self.operator.domain_shape, device=self.device, requires_grad=True))
        R[R < ts.epsilon] = np.inf
        R.reciprocal_()

        itr = tqdm(range(num_iters)) if progress_bar else range(num_iters)
        for _ in itr:
            x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
            x_cur = torch.clone(x_new)
            if min_constraint:
                x_cur[x_cur < 0] = 0

        return x_cur


class Radon3D(nn.Module):
    """
    3D parallel-beam Radon transform with SIRT reconstruction.

    Parameters
    ----------
    depth : int
        Number of detector rows (volume depth along the tilt axis).
    size : int
        Image size in the reconstruction plane (square: size × size).
    angle : int or array-like
        If int N: N uniformly spaced angles in [0, π).
        If array: exact angles in radians.
    device : str
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(self, depth, size=256, angle=np.arange(0., 180., 1.), device='cuda'):
        super().__init__()
        self.img_size = size
        self.angle = angle
        self.depth = depth
        self.device = device
        self._init_op()

    def _init_op(self):
        self.vg = ts.volume(size=1, shape=(self.depth, self.img_size, self.img_size))
        self.pg = ts.parallel(angles=self.angle, shape=(self.depth, self.img_size), size=(1, 1))
        self.operator = ts.operator(self.vg, self.pg)

    def forward(self, img):
        """
        Forward Radon transform (sinogram generation).

        Parameters
        ----------
        img : torch.Tensor, shape (depth, img_size, img_size)

        Returns
        -------
        torch.Tensor, shape (depth, n_angles, img_size)
        """
        return to_autograd(self.operator)(img)

    def bp(self, sino):
        """
        Unfiltered backprojection.

        Parameters
        ----------
        sino : torch.Tensor, shape (depth, n_angles, img_size)

        Returns
        -------
        torch.Tensor, shape (depth, img_size, img_size)
        """
        return to_autograd(self.operator.T)(sino)

    def backward_sirt_ts(self, sino, progress_bar=False, min_constraint=True, num_iters=100):
        """
        SIRT reconstruction from a 3D sinogram.

        Parameters
        ----------
        sino : torch.Tensor, shape (depth, n_angles, img_size)
        progress_bar : bool
            Show tqdm progress bar.
        min_constraint : bool
            Enforce non-negativity at each iteration.
        num_iters : int
            Number of SIRT iterations.

        Returns
        -------
        torch.Tensor, shape (depth, img_size, img_size)
        """
        y = sino.to(self.device)
        x_cur = torch.zeros(self.operator.domain_shape, device=self.device, requires_grad=True)

        C = to_autograd(self.operator.T)(
            torch.ones(self.operator.range_shape, device=self.device, requires_grad=True))
        C[C < ts.epsilon] = np.inf
        C.reciprocal_()

        R = to_autograd(self.operator)(
            torch.ones(self.operator.domain_shape, device=self.device, requires_grad=True))
        R[R < ts.epsilon] = np.inf
        R.reciprocal_()

        itr = tqdm(range(num_iters)) if progress_bar else range(num_iters)
        for _ in itr:
            x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
            x_cur = torch.clone(x_new)
            if min_constraint:
                x_cur[x_cur < 0] = 0

        return x_cur


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def sirt_slice_2d(sinogram, angles_deg, n_iter=30, device=None):
    """
    SIRT reconstruction of a single 2D sinogram slice (numpy in/out).

    Convenience wrapper around ``Radon2D.backward_sirt_ts`` — useful for
    quickly inspecting a slice before running the full 3D reconstruction.

    Parameters
    ----------
    sinogram : np.ndarray, shape (n_angles, img_size)
        Input sinogram for one depth slice.
    angles_deg : array-like
        Tilt angles in degrees, same length as ``n_angles``.
    n_iter : int
        Number of SIRT iterations.
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect CUDA).

    Returns
    -------
    np.ndarray, shape (img_size, img_size)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    _, img_size = sinogram.shape

    rad = Radon2D(size=img_size, angle=angles_rad, device=device)
    sino_t = torch.from_numpy(sinogram.astype(np.float32)).to(device)
    reco = rad.backward_sirt_ts(sino_t, num_iters=n_iter, min_constraint=True)

    return reco.detach().cpu().numpy().squeeze()
