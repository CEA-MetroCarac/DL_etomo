"""
quantification.py
=================
Cliff-Lorimer quantification for EDX tomography volumes.

Implements the classic (hyperspy-style) Cliff-Lorimer formula using direct
intensity ratios.  K-factors come from kfactors_db.py (same dl_etomo/ package).

Public API
----------
get_kfactor_from_line(xray_line)                     -> float
quantify_cl_vol(intensities, xray_lines,
                absorption=None, min_intensity=0.1)  -> list[np.ndarray]

Intensities must be in **physical units** (denormalized) before calling
quantify_cl_vol.  Normalized [0, 1] data will give wrong ratios.

Cliff-Lorimer is a strictly *linear* intensity-ratio law
(C_A/C_B = k_AB * I_A/I_B) -- quantifying directly on log-transformed
counts would silently give wrong compositions, not just noisier ones.
What low-count data (e.g. denoised trace-element maps) needs instead is
numerically *robust* ratio arithmetic: quantify_cl_vol computes each
per-pixel intensity ratio in log-domain (log_stable=True, the default)
-- i.e. exp(log(a) - log(b)) instead of a/b -- which is the exact same
ratio but immune to inf/nan when a channel's intensity is exactly or
nearly zero.
"""

import numpy as np
from functools import reduce
from .kfactors_db import k_factors, atomic_weights as _atomic_weights


# ---------------------------------------------------------------------------
# Weight % → atomic % conversion (no exspy dependency)
# ---------------------------------------------------------------------------

def _weight_to_atomic(weight_percent_list, elements):
    """
    Convert weight fractions to atomic percent.

    Parameters
    ----------
    weight_percent_list : list of np.ndarray  (weight fractions, NOT percent)
    elements : list of str  element symbols

    Returns
    -------
    list of np.ndarray  (atomic percent [0-100], same shapes as input)
    """
    M = np.array([_atomic_weights[el] for el in elements], dtype=float)
    wt = np.stack(weight_percent_list, axis=0)          # (E, *S)

    # at_i = (wt_i / M_i) / Σ_j (wt_j / M_j)  ×  100
    shape = (-1,) + (1,) * (wt.ndim - 1)
    at = wt / M.reshape(shape)
    total = at.sum(axis=0, keepdims=True)
    total = np.where(total == 0.0, 1.0, total)          # avoid division by zero
    at = at / total * 100.0

    return [at[i] for i in range(len(elements))]


# ---------------------------------------------------------------------------
# K-factor lookup
# ---------------------------------------------------------------------------

def get_kfactor_from_line(xray_line):
    """
    Return the k-factor for an X-ray line string such as 'Fe_Ka' or 'Sb_La'.

    Parameters
    ----------
    xray_line : str  e.g. 'Ge_Ka', 'Sb_La', 'Te_La'
    """
    element, line = xray_line.split('_')
    shell = line[0].upper()
    try:
        kf = k_factors[element][shell]
    except KeyError:
        raise ValueError(f"No k-factor found for '{xray_line}'. "
                         f"Check element '{element}' and shell '{shell}' in kfactors_db.py.")
    if kf == 0.0:
        raise ValueError(f"K-factor for '{xray_line}' is 0 in kfactors_db.py — "
                         f"check that the correct shell is used.")
    return kf


# ---------------------------------------------------------------------------
# Classic (hyperspy-style) per-voxel CL
# ---------------------------------------------------------------------------

def _quantification_cliff_lorimer(intensities, kfactors_arr, absorption_correction,
                                   ref_index=0, ref_index2=1, log_stable=True, eps=1e-12):
    """
    Cliff-Lorimer quantification for a single pixel/voxel (direct-ratio version).

    Parameters
    ----------
    intensities : (E,) array
    kfactors_arr : (E,) array
    absorption_correction : (E,) array  values in (0, 1]
    ref_index, ref_index2 : int
    log_stable : bool
        ab[i] = (Ia*Aa*Ka) / (Ii*Ai*Ki) is the same ratio whether computed as
        a direct division or as exp(log(num) - log(den)).  The log-domain form
        (default) is used here because it stays finite (no inf/nan) when a
        channel's absorption-corrected intensity is exactly or near zero --
        common in low-count / denoised EDX maps.  Set False for plain division.
    eps : float
        Numerical floor added before taking logs; keep far below the smallest
        physically meaningful intensity in the data.

    Returns
    -------
    composition : (E,) float  weight fractions.
    """
    if len(intensities) != len(kfactors_arr):
        raise ValueError(
            "The number of kfactors must match the size of the first axis of intensities."
        )

    ab          = np.zeros_like(intensities, dtype=float)
    composition = np.ones_like(intensities,  dtype=float)

    other_index = list(range(len(kfactors_arr)))
    other_index.pop(ref_index)

    num = intensities[ref_index] * absorption_correction[ref_index] * kfactors_arr[ref_index]

    # ab[i] = (Ia * Aa * Ka) / (Ii * Ai * Ki)
    for i in other_index:
        den = intensities[i] * absorption_correction[i] * kfactors_arr[i]
        if log_stable:
            ab[i] = np.exp(np.log(num + eps) - np.log(den + eps))
        else:
            ab[i] = num / den

    # Ca = ab_b / (1 + ab_b + ab_b/ab_c + ...)
    for i in other_index:
        if i == ref_index2:
            composition[ref_index] += ab[ref_index2]
        else:
            composition[ref_index] += ab[ref_index2] / ab[i]

    composition[ref_index] = ab[ref_index2] / composition[ref_index]

    # Ci = Ca / ab[i]
    for i in other_index:
        composition[i] = composition[ref_index] / ab[i]

    return composition


# ---------------------------------------------------------------------------
# Volume-level entry point
# ---------------------------------------------------------------------------

def quantify_cl_vol(intensities_list, xray_lines, absorption=None, min_intensity=0.1,
                    mask=None, auto_mask=False, log_stable=True, eps=1e-12):
    """
    Cliff-Lorimer quantification on ND intensity maps.

    Parameters
    ----------
    intensities_list : array-like, shape (E, *S)
        **Denormalized** intensity maps — one array per element.
        S can be any spatial shape: (Z, Y, X) for 3-D, (Y, X) for 2-D, etc.
    xray_lines : list of str
        X-ray lines in the same order as intensities_list.
        e.g. ['Ge_Ka', 'Sb_La', 'Te_La']
    absorption : array-like, shape (E, *S) or None
        Absorption correction map in (0, 1].  None → no correction (ones).
    min_intensity : float
        Pixels/voxels where all channels are below this threshold are set to
        zero composition.  The two channels with the strongest signal above
        the threshold are used as CL reference indices.  Must be expressed in
        the same physical units as intensities_list (e.g. accumulated counts).
    mask : array-like, shape (*S) or None
        Boolean or float mask applied to every channel before quantification.
        True/1 → keep voxel, False/0 → zero out (background).
        If None and auto_mask=True the mask is computed automatically.
    auto_mask : bool
        If True and mask is None, compute a binary Otsu mask on the sum of
        all intensity channels (same as the reference pruebas workflow).
    log_stable : bool
        Compute the per-pixel Cliff-Lorimer intensity ratios in log-domain
        (default True).  Same physics/result as direct division, but immune
        to inf/nan when a channel's intensity is exactly or near zero --
        the regime low-count / denoised EDX maps often sit in.
    eps : float
        Numerical floor for the log-domain ratio (only used if log_stable).

    Returns
    -------
    atomic_percent : list of np.ndarray, each shape (*S)
        One map per element, in atomic percent [0-100].
    """
    intensities = np.array(intensities_list, dtype=float)
    E = intensities.shape[0]

    if len(xray_lines) != E:
        raise ValueError(f"Got {E} intensity maps but {len(xray_lines)} xray_lines.")

    kfactors_arr = np.array([get_kfactor_from_line(xl) for xl in xray_lines], dtype=float)

    if absorption is None:
        absorption_correction = np.ones_like(intensities)
    else:
        absorption_correction = np.asarray(absorption, dtype=float)
        if absorption_correction.shape != intensities.shape:
            raise ValueError(
                f"absorption shape {absorption_correction.shape} doesn't match "
                f"intensity shape {intensities.shape}."
            )

    # Build and apply mask
    if mask is None and auto_mask:
        from skimage.filters import threshold_otsu
        data_sum = intensities.sum(axis=0)
        th = threshold_otsu(data_sum)
        mask = (data_sum >= th).astype(float)
        print(f'[auto_mask] Otsu threshold={th:.4g}  '
              f'foreground voxels={int(mask.sum())} / {mask.size}')

    if mask is not None:
        mask = np.asarray(mask, dtype=float)
        if mask.shape != intensities.shape[1:]:
            raise ValueError(
                f"mask shape {mask.shape} doesn't match spatial shape "
                f"{intensities.shape[1:]}."
            )
        for i in range(E):
            intensities[i] *= mask
        absorption_correction[:] *= mask[np.newaxis]

    dim    = intensities.shape
    N      = reduce(lambda x, y: x * y, dim[1:])
    intens = intensities.reshape(E, N)
    absorb = absorption_correction.reshape(E, N)

    for i in range(N):
        idx = np.where(intens[:, i] > min_intensity)[0]
        if len(idx) > 1:
            ref_index, ref_index2 = int(idx[0]), int(idx[1])
            intens[:, i] = _quantification_cliff_lorimer(
                intens[:, i], kfactors_arr, absorb[:, i],
                ref_index, ref_index2, log_stable=log_stable, eps=eps,
            )
        else:
            intens[:, i] = np.zeros(E)
            if len(idx) == 1:
                intens[idx[0], i] = 1.0

    intens = intens.reshape(dim)

    atomic_percent = _weight_to_atomic(
        [intens[i] for i in range(E)],
        elements=[xl.split('_')[0] for xl in xray_lines],
    )
    return atomic_percent
