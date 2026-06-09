"""
psd_resolution.py
=================
PSD-based resolution estimation for 3D STEM-EDX-EELS tomographic reconstructions.

Public API
----------
reorder_vol_to_zyx            -- reorder volume to (Z, Y, X) physical convention
make_axis_labels              -- build axis label dict for plots
compute_psd_analysis          -- compute 3D PSD + 1D profiles
plot_real_space               -- orthogonal real-space slices
plot_psd_planes               -- three principal PSD planes
plot_axis_profiles_1d         -- 1D PSD profiles along kx / ky / kz
plot_axis_profiles_overlay    -- all three profiles in one panel
lorentz_psd_C                 -- Lorentzian + pedestal model
fit_lorentz_cutoff            -- resolution fitting
plot_lorentz_fit              -- diagnostic plot for one fit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# Axis mapping
# ─────────────────────────────────────────────────────────────────────────────

def reorder_vol_to_zyx(vol, axis_map):
    """
    Transpose *vol* to physical (Z, Y, X) order.

    Parameters
    ----------
    vol : ndarray
        Input volume with arbitrary axis ordering.
    axis_map : dict {int -> str}
        Maps each axis index (0, 1, 2) to its physical label ("X", "Y", or "Z").
        Z is the tilt / missing-wedge axis.

    Returns
    -------
    vol_zyx : ndarray
        Volume transposed to (Z, Y, X) order.
    perm : list of int
        Permutation applied: ``vol_zyx = np.transpose(vol, perm)``.
    """
    assert set(axis_map.values()) == {"X", "Y", "Z"}
    assert set(axis_map.keys())   == {0, 1, 2}
    perm = [next(k for k,v in axis_map.items() if v == c) for c in "ZYX"]
    vol_zyx = np.transpose(vol, perm)
    print(f"[reorder] {perm}: {vol.shape} -> ZYX {vol_zyx.shape}")
    return vol_zyx, perm


def make_axis_labels(units=r"\mathrm{nm}^{-1}"):
    """
    Build the axis-label dict used by all plot functions.

    Parameters
    ----------
    units : str, optional
        LaTeX string for the frequency unit; default ``r"\\mathrm{nm}^{-1}"``.

    Returns
    -------
    dict
        Keys: ``kx``, ``ky``, ``kz`` (full axis labels with units),
        ``kx_title``, ``ky_title``, ``kz_title`` (short panel title strings).
    """
    return {
        "kx":       rf"$k_{{X}}\;({units})$",
        "ky":       rf"$k_{{Y}}\;({units})$",
        "kz":       rf"$k_{{Z\,\mathrm{{tilt}}}}\;({units})$",
        "kx_title": r"$k_X$",
        "ky_title": r"$k_Y$",
        "kz_title": r"$k_{Z\,\mathrm{tilt}}$",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PSD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _moving_median(x, w):
    """
    Edge-padded running median.

    Parameters
    ----------
    x : array_like
        1-D input signal.
    w : int
        Window width; forced to the nearest odd integer.

    Returns
    -------
    ndarray
        Smoothed signal, same length as *x*.
    """
    x = np.asarray(x, float)
    w = int(w) | 1          # ensure odd
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i+w]) for i in range(len(x))])


def _norm(a):
    """
    Normalise array to [0, 1] using the 1st–99th percentile range.

    Parameters
    ----------
    a : ndarray
        Input array (any shape); non-finite values are ignored for the
        percentile computation but are propagated in the output.

    Returns
    -------
    ndarray
        Clipped normalised array, same shape as *a*.
    """
    lo, hi = np.percentile(a[np.isfinite(a)], [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _axis_profile(PSD, axis, band):
    """
    Average a ±band central slab of the 3D PSD along one axis.

    Parameters
    ----------
    PSD : ndarray, shape (Z, Y, X)
        3D power spectral density array (fftshifted).
    axis : {"kx", "ky", "kz"}
        Axis along which to extract the 1-D profile.
    band : int
        Half-width (in voxels) of the slab averaged perpendicular to *axis*.

    Returns
    -------
    ndarray, shape (N,)
        Mean 1-D profile along the requested axis.
    """
    Z, Y, X = PSD.shape
    z0, y0, x0 = Z//2, Y//2, X//2
    sl = lambda c, n: slice(max(0, c-band), min(n, c+band+1))
    if axis == "kx": return PSD[sl(z0,Z), sl(y0,Y), :].mean(axis=(0,1))
    if axis == "ky": return PSD[sl(z0,Z), :, sl(x0,X)].mean(axis=(0,2))
    if axis == "kz": return PSD[:, sl(y0,Y), sl(x0,X)].mean(axis=(1,2))
    raise ValueError(axis)


def _quick_cutoff(k, psd, *, tail_frac, smooth_w, tol_factor,
                  k_min_frac, noise_mode, mad_sigma, counts=None):
    """
    Find the resolution cutoff as the first descending crossing of the
    smoothed PSD with a noise threshold.

    Parameters
    ----------
    k : ndarray
        Spatial-frequency axis (positive half, ascending).
    psd : ndarray
        Raw 1-D PSD values corresponding to *k*.
    tail_frac : float
        Fraction of the profile (from the high-k end) used to estimate noise.
    smooth_w : int
        Running-median window width passed to ``_moving_median``.
    tol_factor : float
        Threshold multiplier applied to the noise estimate.
    k_min_frac : float
        Low-frequency guard: crossings below ``k_min_frac * k[-1]`` are ignored.
    noise_mode : {"p80", "mad"}
        Noise estimator: 80th percentile of the tail, or median + MAD.
    mad_sigma : float
        MAD multiplier (used only when ``noise_mode="mad"``).
    counts : array_like or None, optional
        If provided, positions where counts ≤ 0 are excluded from the noise tail.

    Returns
    -------
    kc : float
        Cutoff frequency (nm⁻¹); ``np.nan`` if no crossing is found.
    noise : float
        Estimated noise level.
    thr : float
        Threshold value (``tol_factor * noise``).
    sm : ndarray
        Smoothed PSD.
    kguard : float
        Low-frequency guard value (``k_min_frac * k[-1]``).
    i0 : int
        Index where the noise tail starts.
    """
    sm = _moving_median(psd, smooth_w)
    n  = len(sm)
    i0 = max(1, int((1 - tail_frac) * n))
    valid = np.isfinite(sm) & (sm > 0)
    if counts is not None:
        valid &= np.asarray(counts) > 0
    tail = sm[i0:][valid[i0:]]
    if tail.size < 5:
        noise = float(np.nanmedian(sm[i0:]))
    elif noise_mode == "p80":
        noise = float(np.percentile(tail, 80))
    else:   # mad
        med = float(np.median(tail))
        noise = med + mad_sigma * 1.4826 * float(np.median(np.abs(tail-med)))
    thr    = tol_factor * noise
    kguard = k_min_frac * k[-1]
    s = sm - thr
    for i in range(max(1, int(np.searchsorted(k, kguard))), n):
        if np.isfinite(s[i-1]) and np.isfinite(s[i]) and s[i-1]>0 and s[i]<=0:
            k1,k2,s1,s2 = k[i-1],k[i],s[i-1],s[i]
            kc = k2 if s1==s2 else k1 + (0-s1)*(k2-k1)/(s2-s1)
            return float(kc), noise, thr, sm, float(kguard), i0
    return np.nan, noise, thr, sm, float(kguard), i0


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_psd_analysis(
    vol_zyx,
    voxel_nm=(0.4, 0.4, 0.4),
    *,
    band=2,
    tail_frac_axis=(0.15, 0.15, 0.15),
    tol_factor=1.0,
    smooth_w=11,
    k_min_frac=0.10,
    noise_mode="p80",
    mad_sigma=3.0,
):
    """
    Compute the 3D PSD of *vol_zyx* and extract 1D profiles along kx/ky/kz.

    Parameters
    ----------
    vol_zyx : ndarray, shape (Z, Y, X)
        Reconstruction volume; Z must be the tilt / missing-wedge axis.
    voxel_nm : tuple of float, (dz, dy, dx)
        Voxel size in nm along each axis.
    band : int
        Half-width (in voxels) of the central slab averaged for each 1D profile.
    tail_frac_axis : tuple of float, (fx, fy, fz)
        Tail fraction for the quick noise cutoff on each axis.
    tol_factor : float
        Threshold multiplier on the noise estimate (``thr = tol_factor * noise``).
    smooth_w : int
        Running-median window width for profile smoothing.
    k_min_frac : float
        Low-frequency guard as a fraction of the Nyquist frequency.
    noise_mode : {"p80", "mad"}
        Noise estimator: 80th percentile or median ± MAD.
    mad_sigma : float
        MAD multiplier (only used when ``noise_mode="mad"``).

    Returns
    -------
    dict with keys:
        vol_shape : tuple (Z, Y, X)
        centers   : dict {z0, y0, x0}  — central slice indices
        voxel_nm  : tuple (dz, dy, dx)
        PSD       : ndarray, shape (Z, Y, X)
        axes      : dict {fx, fy, fz, dkx, dky, dkz, nyquist}
        planes    : dict {kxky, kxkz, kykz}  — 2-D central PSD slices
        axis_1d   : dict {kx, ky, kz}  — each a dict with k, raw, smooth,
                    kc, noise, thr, kmin, i0
    """
    def _to3(x):
        return (float(x),)*3 if np.isscalar(x) else tuple(float(v) for v in x)

    tf = _to3(tail_frac_axis)
    dz, dy, dx = map(float, voxel_nm)
    Z, Y, X = vol_zyx.shape
    z0, y0, x0 = Z//2, Y//2, X//2

    PSD = np.abs(np.fft.fftshift(np.fft.fftn(vol_zyx.astype(np.float32))))**2

    fx = np.fft.fftshift(np.fft.fftfreq(X, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Y, d=dy))
    fz = np.fft.fftshift(np.fft.fftfreq(Z, d=dz))
    dkx = abs(fx[1]-fx[0]); dky = abs(fy[1]-fy[0]); dkz = abs(fz[1]-fz[0])

    def _profile(axis, freq, tf_i):
        prof = _axis_profile(PSD, axis, band)
        m = freq >= 0
        kp, pp = freq[m], prof[m]
        kc, noise, thr, sm, kmin, i0 = _quick_cutoff(
            kp, pp,
            tail_frac=tf_i, smooth_w=smooth_w, tol_factor=tol_factor,
            k_min_frac=k_min_frac, noise_mode=noise_mode, mad_sigma=mad_sigma)
        return {"k":kp, "raw":pp, "smooth":sm, "kc":kc,
                "noise":noise, "thr":thr, "kmin":kmin, "i0":i0}

    return {
        "vol_shape": (Z, Y, X),
        "centers":   {"z0":z0, "y0":y0, "x0":x0},
        "voxel_nm":  (dz, dy, dx),
        "PSD":       PSD,
        "axes": {
            "fx":fx, "fy":fy, "fz":fz,
            "dkx":dkx, "dky":dky, "dkz":dkz,
            "nyquist": (1/(2*dx), 1/(2*dy), 1/(2*dz)),
        },
        "planes": {
            "kxky": PSD[z0, :, :],   # shape (Y, X)
            "kxkz": PSD[:, y0, :],   # shape (Z, X)
            "kykz": PSD[:, :, x0],   # shape (Z, Y)
        },
        "axis_1d": {
            "kx": _profile("kx", fx, tf[0]),
            "ky": _profile("ky", fy, tf[1]),
            "kz": _profile("kz", fz, tf[2]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_real_space(vol_zyx, centers, cmap, voxel_nm=(1,1,1), title_prefix="Real space"):
    """
    Show three orthogonal central slices of the volume in physical nm units.

    Parameters
    ----------
    vol_zyx : ndarray, shape (Z, Y, X)
        Reconstruction volume.
    centers : dict {z0, y0, x0}
        Integer indices of the central slices; falls back to shape//2 if a key
        is missing.
    cmap : str or Colormap
        Matplotlib colormap for the images.
    voxel_nm : tuple of float, (dz, dy, dx), optional
        Voxel size in nm; default (1, 1, 1).
    title_prefix : str, optional
        Figure suptitle; default "Real space".

    Returns
    -------
    matplotlib.figure.Figure
    """
    dz, dy, dx = voxel_nm
    Z, Y, X = vol_zyx.shape
    z0 = centers.get("z0", Z//2)
    y0 = centers.get("y0", Y//2)
    x0 = centers.get("x0", X//2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    kw = dict(cmap=cmap, origin="lower", aspect="equal")
    ax[0].imshow(_norm(vol_zyx[z0,:,:]), extent=[0,X*dx,0,Y*dy], **kw)
    ax[0].set(title="XY", xlabel="x (nm)", ylabel="y (nm)")
    ax[1].imshow(_norm(vol_zyx[:,y0,:]), extent=[0,X*dx,0,Z*dz], **kw)
    ax[1].set(title="XZ", xlabel="x (nm)", ylabel="z (nm)")
    ax[2].imshow(_norm(vol_zyx[:,:,x0]), extent=[0,Y*dy,0,Z*dz], **kw)
    ax[2].set(title="YZ", xlabel="y (nm)", ylabel="z (nm)")
    plt.suptitle(title_prefix, y=1.01)
    plt.tight_layout(); plt.show()
    return fig


def plot_psd_planes(out, vmin_log=None, vmax_log=None, cmap="gray", axis_labels=None, eps=1e-30):
    """
    Show the three principal PSD planes (kx–ky, kx–kz, ky–kz) on a log scale.

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    vmin_log : float or None
        Lower display limit in log₁₀ units; auto-set to the 10th percentile if None.
    vmax_log : float or None
        Upper display limit in log₁₀ units; auto-set to the 99.99th percentile if None.
    cmap : str or Colormap, optional
        Matplotlib colormap; default "gray".
    axis_labels : dict or None, optional
        Label dict from ``make_axis_labels``; built automatically if None.
    eps : float, optional
        Floor added before log₁₀ to avoid log(0); default 1e-30.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fx, fy, fz = out["axes"]["fx"], out["axes"]["fy"], out["axes"]["fz"]
    lbls = axis_labels or make_axis_labels()

    Ls = [np.log10(np.maximum(out["planes"][k], eps))
          for k in ("kxky","kxkz","kykz")]
    all_v = np.concatenate([L.ravel() for L in Ls])
    if vmax_log is None:
        vmax_log = float(np.nanpercentile(all_v, 99.99))
    if vmin_log is None:
        vmin_log = float(np.nanpercentile(all_v, 10.0))

    exts   = [[fx[0],fx[-1],fy[0],fy[-1]],
               [fx[0],fx[-1],fz[0],fz[-1]],
               [fy[0],fy[-1],fz[0],fz[-1]]]
    xlbls  = [lbls["kx"], lbls["kx"], lbls["ky"]]
    ylbls  = [lbls["ky"], lbls["kz"], lbls["kz"]]
    titles = [f"{lbls['kx_title']}–{lbls['ky_title']}",
              f"{lbls['kx_title']}–{lbls['kz_title']}  ← MW",
              f"{lbls['ky_title']}–{lbls['kz_title']}  ← MW"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    kw = dict(cmap=cmap, vmin=vmin_log, vmax=vmax_log, origin="lower", aspect="auto")
    for ax, L, ext, xl, yl, t in zip(axes, Ls, exts, xlbls, ylbls, titles):
        im = ax.imshow(L, extent=ext, **kw)
        ax.set(xlabel=xl, ylabel=yl, title=t)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()
    return fig


def plot_axis_profiles_1d(out, axis_labels=None, show_tail=True):
    """
    Plot 1D PSD profiles along kx, ky, and kz with threshold and cutoff markers.

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    axis_labels : dict or None, optional
        Label dict from ``make_axis_labels``; built automatically if None.
    show_tail : bool, optional
        Shade the noise-tail region used to estimate the threshold; default True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lbls = axis_labels or make_axis_labels()
    nyq  = out["axes"]["nyquist"]
    cfg  = [("kx", nyq[0], lbls["kx"], lbls["kx_title"]),
            ("ky", nyq[1], lbls["ky"], lbls["ky_title"]),
            ("kz", nyq[2], lbls["kz"], lbls["kz_title"])]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for ax, (key, nyq_i, xlabel, title_k) in zip(axes, cfg):
        d   = out["axis_1d"][key]
        k   = np.asarray(d["k"]); raw = np.asarray(d["raw"])
        sm  = np.asarray(d["smooth"])
        kc  = float(d["kc"]); thr = float(d["thr"]); i0 = int(d["i0"])

        ax.scatter(k, raw, s=10, alpha=0.3, label="PSD")
        ax.plot(k, sm, lw=1, color="orange", label="smoothed")
        if show_tail and 0 <= i0 < len(k):
            ax.axvspan(k[i0], k[-1], alpha=0.12, label="noise tail")
        ax.axhline(thr, ls="--", lw=1, label="threshold")
        if np.isfinite(kc) and kc > 0:
            ax.axvline(kc, ls=":", lw=2,
                       label=f"kc={kc:.3g} → {1/kc:.2f} nm")
        ax.axvline(nyq_i, ls="--", lw=1, color="k")
        ax.text(0.02, 0.95, f"Nyquist={nyq_i:.3g} nm⁻¹",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set(yscale="log", title=f"PSD along {title_k}",
               xlabel=xlabel, ylabel="PSD (a.u.)")
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout(); plt.show()
    return fig


def plot_axis_profiles_overlay(out, axis_labels=None):
    """
    Plot all three 1D profiles in one panel to identify the fastest-dropping axis.

    The axis whose PSD drops fastest toward high frequencies is typically the
    tilt (missing-wedge) axis, which limits resolution most strongly.

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    axis_labels : dict or None, optional
        Label dict from ``make_axis_labels``; built automatically if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lbls = axis_labels or make_axis_labels()
    cfg  = [("kx", lbls["kx_title"], "tab:blue"),
            ("ky", lbls["ky_title"], "tab:orange"),
            ("kz", lbls["kz_title"], "tab:red")]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, title, color in cfg:
        d  = out["axis_1d"][key]
        k  = np.asarray(d["k"]); sm = np.asarray(d["smooth"])
        kc = float(d["kc"])
        ax.plot(k, sm, lw=1.5, color=color, label=title)
        if np.isfinite(kc) and kc > 0:
            ax.axvline(kc, ls=":", lw=1.5, color=color,
                       label=f"kc={kc:.3g} → {1/kc:.2f} nm")
    ax.set(yscale="log", xlabel=r"$k\;(\mathrm{nm}^{-1})$",
           ylabel="PSD (a.u.)",
           title="Overlaid profiles — fastest drop = tilt axis")
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Lorentzian resolution fitting
# ─────────────────────────────────────────────────────────────────────────────

def lorentz_psd_C(k, A, xi, p, C):
    """
    Lorentzian PSD model with additive pedestal.

    .. math::

        \\mathrm{PSD}(k) = \\frac{A}{\\left(1 + (2\\pi k \\xi)^2\\right)^p} + C

    Parameters
    ----------
    k : array_like
        Spatial frequency (nm⁻¹).
    A : float
        Signal amplitude at k = 0, above the pedestal.
    xi : float
        Correlation length (nm); controls the roll-off frequency.
    p : float
        Decay exponent; p = 1 gives a standard Lorentzian.
    C : float
        Additive noise pedestal.

    Returns
    -------
    ndarray
        Model PSD values at each *k*.
    """
    k = np.asarray(k, float)
    return A / (1.0 + (2.0*np.pi*k*xi)**2)**p + C


def fit_lorentz_cutoff(
    k, psd_s,
    *,
    kmin=0.01,
    kmax_frac=0.97,
    min_pts=20,
    noise_tail_frac=0.20,
    noise_kmax_frac=1.0,
    choose_cut="last",
    eps=1e-30,
):
    """
    Fit a Lorentzian+C model to a 1D PSD profile and find the resolution cutoff.

    The cutoff is defined as the crossing of the fitted Lorentzian curve with a
    linear noise floor estimated from the high-frequency tail of the profile
    (in log space).

    Parameters
    ----------
    k : array_like
        Spatial-frequency axis (nm⁻¹), positive half only.
    psd_s : array_like
        Smoothed 1-D PSD values corresponding to *k*.
    kmin : float, optional
        Lower bound of the Lorentzian fit window (nm⁻¹); default 0.01.
    kmax_frac : float, optional
        Upper bound as a fraction of the maximum valid frequency; default 0.97.
    min_pts : int, optional
        Minimum number of points required inside the fit window; default 20.
    noise_tail_frac : float, optional
        Fraction of the full profile (from the high-k end) used to fit the
        noise line in log space; default 0.20.
    noise_kmax_frac : float, optional
        Trim the profile above ``noise_kmax_frac * kmax`` before the noise fit
        to guard against high-frequency artefacts; default 1.0 (no trim).
    choose_cut : {"first", "last"}, optional
        Which crossing to report when multiple are found; default "last".
    eps : float, optional
        Numerical floor added before log₁₀ to avoid log(0); default 1e-30.

    Returns
    -------
    dict
        Always contains key ``ok`` (bool).

        On success (``ok=True``):
            ``A``, ``xi``, ``p``, ``C``         — fit parameters,
            ``A_err``, ``xi_err``, ``p_err``, ``C_err`` — 1-σ uncertainties,
            ``r2_lorentz``                       — R² of the Lorentzian fit,
            ``fit_range``                        — (kmin, khi) tuple,
            ``noise_line``                       — dict {a, b, r2, n_tail,
                                                   k_range, k_used, psd_used},
            ``k_cut``, ``res_cut_nm``            — cutoff frequency and resolution,
            ``cut_mode``                         — "crossing" or "min_distance",
            ``crossings``                        — list of all crossing frequencies,
            ``log_gap_at_cut``                   — log₁₀ gap at the chosen cutoff.

        On failure (``ok=False``):
            ``reason`` — description string,
            ``fit_range`` — present when the window was established before failure.
    """
    k     = np.asarray(k, float)
    psd_s = np.asarray(psd_s, float)

    base = np.isfinite(k) & np.isfinite(psd_s) & (psd_s > 0) & (k >= kmin)
    if base.sum() < min_pts:
        return {"ok": False, "reason": f"too few points above kmin={kmin}"}

    kmax_data = float(np.nanmax(k[base]))
    khi = kmax_frac * kmax_data
    m   = base & (k <= khi)
    if m.sum() < min_pts:
        return {"ok": False, "reason": f"too few pts in [{kmin:.3g},{khi:.3g}]",
                "fit_range": (kmin, khi)}

    x = k[m]; y = psd_s[m]
    ylog = np.log10(np.maximum(y, eps))

    # initial guesses
    C0    = max(float(np.median(y[-max(8, y.size//6):])), eps)
    A0    = max(float(np.max(y) - C0), float(np.max(y)) * 0.5)
    y0    = np.maximum(y - C0, eps)
    kh    = float(x[np.argmin(np.abs(y0 - np.max(y0)/2))])
    xi0   = 1.0 / (2*np.pi*kh + 1e-30) if kh > 0 else 1.0
    C_min = max(float(np.median(y[-max(5, x.size//10):])), eps)

    def _lm(kv, A, xi, p, C):
        return np.log10(np.maximum(lorentz_psd_C(kv, A, xi, p, C), eps))

    try:
        popt, pcov = curve_fit(_lm, x, ylog, p0=[A0, xi0, 1.2, C0],
                               bounds=([eps,eps,0.05,C_min],[np.inf,np.inf,20,np.inf]),
                               maxfev=80000)
    except Exception as e:
        return {"ok": False, "reason": str(e), "fit_range": (kmin, khi)}

    A, xi, p, C = map(float, popt)
    try:
        perr = np.sqrt(np.diag(pcov))
        A_e, xi_e, p_e, C_e = map(float, perr)
    except (ValueError, np.linalg.LinAlgError):
        A_e = xi_e = p_e = C_e = np.nan

    yh = _lm(x, A, xi, p, C)
    r2 = 1 - float(np.sum((ylog-yh)**2)) / (float(np.sum((ylog-ylog.mean())**2)) + 1e-30)

    # noise line on the full valid profile (not limited by kmax_frac)
    bn    = base & (k <= noise_kmax_frac * kmax_data)
    order = np.argsort(k[bn])
    ks, ys = k[bn][order], psd_s[bn][order]
    n_tail = max(8, int(np.ceil(noise_tail_frac * ks.size)))
    kt, pt = ks[-n_tail:], ys[-n_tail:]
    yt = np.log10(np.maximum(pt, eps))
    b_n, a_n = np.polyfit(kt, yt, 1)
    r2n = 1 - float(np.sum((yt - (a_n+b_n*kt))**2)) / (float(np.sum((yt-yt.mean())**2))+1e-30)

    # crossings (any sign change) or min distance
    ke = np.sort(k[np.isfinite(k) & (k >= kmin)])
    lg = (np.log10(np.maximum(lorentz_psd_C(ke,A,xi,p,C), eps))
          - np.log10(np.maximum(10**(a_n+b_n*ke), eps)))

    crosses = []
    for i in range(1, ke.size):
        if np.isfinite(lg[i-1]) and np.isfinite(lg[i]) and lg[i-1]*lg[i] <= 0:
            g1,g2,k1,k2 = lg[i-1],lg[i],ke[i-1],ke[i]
            crosses.append(float(k2) if g1==g2 else float(k1 - g1*(k2-k1)/(g2-g1)))

    if crosses:
        kc = crosses[0] if choose_cut=="first" else crosses[-1]
        mode, gap = "crossing", 0.0
    else:
        idx  = int(np.argmin(np.abs(lg)))
        kc   = float(ke[idx]); gap = float(lg[idx]); mode = "min_distance"

    res_nm = float(1/kc) if np.isfinite(kc) and kc > 0 else np.nan

    return {
        "ok": True,
        "A":A, "xi":xi, "p":p, "C":C,
        "A_err":A_e, "xi_err":xi_e, "p_err":p_e, "C_err":C_e,
        "r2_lorentz": r2,
        "fit_range": (kmin, khi),
        "noise_line": {
            "a":a_n, "b":b_n, "r2":r2n,
            "n_tail":int(kt.size),
            "k_range":(float(kt[0]),float(kt[-1])),
            "k_used":kt.copy(), "psd_used":pt.copy(),
        },
        "k_cut":kc, "res_cut_nm":res_nm,
        "cut_mode":mode, "crossings":crosses, "log_gap_at_cut":gap,
    }


def plot_lorentz_fit(k, d_axis, fit, title="", save_path=None):
    """
    Diagnostic plot for one Lorentzian resolution fit.

    Overlays the raw PSD, the fitted Lorentzian+C curve, the noise line, and
    the resolution cutoff marker on a single log-scale panel.

    Parameters
    ----------
    k : ndarray
        Spatial-frequency axis (nm⁻¹), same grid used for the fit.
    d_axis : dict
        One entry of ``out["axis_1d"]`` (must contain key ``"raw"``).
    fit : dict
        Output of ``fit_lorentz_cutoff`` for this axis.
    title : str, optional
        Panel title string; default "".
    save_path : str or None, optional
        If given, saves the figure to this path at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(k, np.asarray(d_axis["raw"], float), s=12, alpha=0.4, label="PSD raw")

    if fit.get("ok"):
        A, xi, p, C = fit["A"], fit["xi"], fit["p"], fit["C"]
        lo, hi = fit["fit_range"]
        mf = (k >= lo) & (k <= hi)
        ax.plot(k[mf], lorentz_psd_C(k[mf],A,xi,p,C), "r-", lw=1.8,
                label=f"Lorentz+C  R²={fit['r2_lorentz']:.3f}")
        ax.axhline(C, ls=":", lw=1.2, color="tomato", label=f"C={C:.2e}")

        nl = fit["noise_line"]
        ax.plot(k, 10**(nl["a"]+nl["b"]*k), "k--", lw=1.2,
                label=f"Noise R²={nl['r2']:.2f}")
        ax.scatter(nl["k_used"], nl["psd_used"], s=35, color="k", zorder=5,
                   label=f"noise pts (n={nl['n_tail']})")

        for xc in fit.get("crossings",[])[:-1]:
            ax.axvline(xc, color="gray", ls="--", lw=0.8, alpha=0.5)

        kc = fit["k_cut"]
        if np.isfinite(kc) and kc > 0:
            lbl = f"kc={kc:.3g} nm⁻¹ → {fit['res_cut_nm']:.2f} nm  [{fit['cut_mode']}]"
            if fit["cut_mode"] == "min_distance":
                lbl += f"  gap={fit['log_gap_at_cut']:+.2f}"
            ax.axvline(kc, color="purple", ls="-.", lw=2, label=lbl)
        else:
            ax.text(0.5, 0.5, "No cutoff", transform=ax.transAxes,
                    ha="center", color="red")

        xi_e = fit.get("xi_err", np.nan)
        if np.isfinite(xi_e):
            ax.text(0.97, 0.97, f"ξ={xi*1e3:.1f}±{xi_e*1e3:.1f} pm",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="darkred")
    else:
        ax.text(0.5, 0.5, f"Fit failed:\n{fit.get('reason','')}",
                transform=ax.transAxes, ha="center", color="red")

    ax.set(yscale="log", xlabel=r"Spatial frequency (nm$^{-1}$)",
           ylabel="PSD (a.u.)", title=title)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
