"""
cs_tv.py
========
CS-TV reconstruction via Condat-Vu primal-dual splitting.

    min_x  ½‖Rx - y‖²  +  λ‖Lx‖₁    s.t. x ≥ 0

Delegates to pysap-etomo (ERadon2D/3D, HOTV/HOTV_3D) and modopt (condatvu).

Public API
----------
suggest_lambda(data, n, center_scale, spread)
    Auto-select a λ sweep range from the data magnitude.

compress_sensing_2d(sinogram_2d, angles, lam, n_iter, ...)
    Condat-Vu on a single 2-D sinogram slice.  Use for lambda selection.

compress_sensing(sinograms, angles, lam, n_iter, ...)
    Condat-Vu on the full 3-D sinogram (true 3-D TV — slices are coupled).
"""

import numpy as np

from etomo.operators import Radon2D as ERadon2D, Radon3D as ERadon3D, HOTV, HOTV_3D
from etomo.reconstructors.forwardtomo import TomoReconstructor
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold


def suggest_lambda(data, n=5, center_scale=1.0, spread=1.5):
    """
    Return n log-spaced λ values centred on the mean positive signal.

    λ should live near the data magnitude — this gives a sweep around that point.

    Parameters
    ----------
    data         : np.ndarray — sinogram (any shape)
    n            : int        — number of values to return
    center_scale : float      — multiplier on the mean (default 1.0)
    spread       : float      — half-decades either side of centre (default 1.5)

    Returns
    -------
    lambdas : list of float, length n, sorted ascending
    """
    data = np.asarray(data, dtype=np.float32)
    pos = data[data > 0]
    center = float(np.mean(pos)) * center_scale if pos.size else 1.0
    return list(np.logspace(-spread, spread, n) * center)



def compress_sensing_2d(
    sinogram_2d,
    angles,
    lam,
    n_iter,
    n_power = 15,   # kept for API compatibility
    device  = None, # kept for API compatibility
    verbose = True,
):
    """
    CS-TV 2-D reconstruction (Condat-Vu primal-dual, analysis form).

    Parameters
    ----------
    sinogram_2d : np.ndarray, shape (n_angles, img_size)
    angles      : array-like — tilt angles in degrees
    lam         : float — TV regularisation weight (larger → smoother)
    n_iter      : int   — Condat-Vu iterations
    verbose     : bool  — print progress

    Returns
    -------
    reco  : np.ndarray, shape (img_size, img_size)
    costs : list of float
    """
    sinogram_2d = np.asarray(sinogram_2d, dtype=np.float32)
    _, img_size = sinogram_2d.shape
    angles_arr  = np.asarray(angles, dtype=np.float32)

    radon_op  = ERadon2D(angles=angles_arr, img_size=img_size, gpu=True, normalized=False)
    linear_op = HOTV([img_size, img_size], order=1)
    reg_op    = SparseThreshold(linear=Identity(), weights=lam)

    reconstructor = TomoReconstructor(
        data_op=radon_op, linear_op=linear_op, regularizer_op=reg_op,
        gradient_formulation='analysis', verbose=int(verbose),
    )
    reco, costs, _ = reconstructor.reconstruct(
        data=sinogram_2d, optimization_alg='condatvu',
        num_iterations=n_iter, cost_op_kwargs={'cost_interval': 1},
    )
    return np.asarray(reco, dtype=np.float32), costs


def compress_sensing(
    sinograms,
    angles,
    lam,
    n_iter,
    n_power = 15,   # kept for API compatibility
    device  = None, # kept for API compatibility
    verbose = True,
):
    """
    CS-TV 3-D reconstruction (Condat-Vu primal-dual, analysis form, true 3-D TV).

    Parameters
    ----------
    sinograms : np.ndarray, shape (depth, n_angles, img_size)
    angles    : array-like — tilt angles in degrees
    lam       : float — TV regularisation weight (larger → smoother)
    n_iter    : int   — Condat-Vu iterations
    verbose   : bool  — print progress

    Returns
    -------
    reco  : np.ndarray, shape (depth, img_size, img_size)
    costs : list of float
    """
    sinograms  = np.asarray(sinograms, dtype=np.float32)
    depth, _, img_size = sinograms.shape
    angles_arr = np.asarray(angles, dtype=np.float32)

    radon_op  = ERadon3D(angles=angles_arr, img_size=img_size, nb_slices=depth, normalized=False)
    linear_op = HOTV_3D(img_shape=[img_size, img_size], nb_slices=depth, order=1)
    reg_op    = SparseThreshold(linear=Identity(), weights=lam)

    reconstructor = TomoReconstructor(
        data_op=radon_op, linear_op=linear_op, regularizer_op=reg_op,
        gradient_formulation='analysis', verbose=int(verbose),
    )
    reco, costs, _ = reconstructor.reconstruct(
        data=sinograms, optimization_alg='condatvu',
        num_iterations=n_iter, cost_op_kwargs={'cost_interval': 1},
    )
    return np.asarray(reco, dtype=np.float32), costs
