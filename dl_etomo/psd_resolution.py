"""
psd_resolution.py
=================
PSD-based resolution estimation for 3D tomographic reconstructions.

After ``compute_psd_analysis`` the axis convention is always:
    dim 0 = kz = physical Z (tilt / missing-wedge axis, fastest PSD decay)
    dim 1 = ky = physical Y
    dim 2 = kx = physical X

Public API
----------
compute_psd_analysis  -- 3D PSD + auto axis detection + 1D profiles
plot_real_space       -- orthogonal real-space slices
plot_psd_planes       -- PSD planes (optional profile-band overlay)
lorentz_psd_C         -- Lorentzian + pedestal model
fit_lorentz_cutoff    -- Lorentzian resolution fitting
plot_lorentz_fit      -- diagnostic plot for one fit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _moving_median(x, w):
    x = np.asarray(x, float)
    w = int(w) | 1
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def _norm(a):
    lo, hi = np.percentile(a[np.isfinite(a)], [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _axis_profile(PSD, axis, band_k, dkx, dky, dkz):
    """
    Extract a 1D PSD profile along *axis* by averaging the central slab
    (width ±band_k in nm⁻¹) in the two perpendicular directions.

    PSD shape is (Z, Y, X):  dim0 ↔ kz,  dim1 ↔ ky,  dim2 ↔ kx.

    kx profile  →  average over Z-slab (±bz) and Y-slab (±by),  result shape (X,)
    ky profile  →  average over Z-slab (±bz) and X-slab (±bx),  result shape (Y,)
    kz profile  →  average over Y-slab (±by) and X-slab (±bx),  result shape (Z,)
    """
    Z, Y, X = PSD.shape
    z0, y0, x0 = Z // 2, Y // 2, X // 2
    bx = max(1, int(np.round(band_k / dkx)))
    by = max(1, int(np.round(band_k / dky)))
    bz = max(1, int(np.round(band_k / dkz)))

    if axis == "kx":   # profile along dim2 (X)
        return PSD[z0 - bz:z0 + bz + 1, y0 - by:y0 + by + 1, :].mean(axis=(0, 1))
    if axis == "ky":   # profile along dim1 (Y)
        return PSD[z0 - bz:z0 + bz + 1, :, x0 - bx:x0 + bx + 1].mean(axis=(0, 2))
    if axis == "kz":   # profile along dim0 (Z / tilt)
        return PSD[:, y0 - by:y0 + by + 1, x0 - bx:x0 + bx + 1].mean(axis=(1, 2))
    raise ValueError(f"Unknown axis: {axis!r}")


def _quick_cutoff(k, psd, *, tail_frac, smooth_w, tol_factor,
                  k_min_frac, noise_mode, mad_sigma):
    """Threshold-crossing cutoff on a smoothed 1D PSD."""
    sm = _moving_median(psd, smooth_w)
    n  = len(sm)
    i0 = max(1, int((1 - tail_frac) * n))
    tail = sm[i0:][np.isfinite(sm[i0:]) & (sm[i0:] > 0)]
    if tail.size < 5:
        noise = float(np.nanmedian(sm[i0:]))
    elif noise_mode == "p80":
        noise = float(np.percentile(tail, 80))
    else:
        med   = float(np.median(tail))
        noise = med + mad_sigma * 1.4826 * float(np.median(np.abs(tail - med)))
    thr    = tol_factor * noise
    kguard = k_min_frac * k[-1]
    s = sm - thr
    for i in range(max(1, int(np.searchsorted(k, kguard))), n):
        if np.isfinite(s[i - 1]) and np.isfinite(s[i]) and s[i - 1] > 0 and s[i] <= 0:
            k1, k2, s1, s2 = k[i - 1], k[i], s[i - 1], s[i]
            kc = k2 if s1 == s2 else k1 + (0 - s1) * (k2 - k1) / (s2 - s1)
            return float(kc), noise, thr, sm, float(kguard), i0
    return np.nan, noise, thr, sm, float(kguard), i0


def _decay_rate(k, sm):
    """Mean log-slope of a smoothed profile (higher = faster decay = worse resolution)."""
    k  = np.asarray(k,  float)
    sm = np.asarray(sm, float)
    m  = np.isfinite(k) & np.isfinite(sm) & (sm > 0) & (k > 0)
    if m.sum() < 5:
        return np.nan
    with np.errstate(divide="ignore"):
        logp = np.log(sm[m] + 1e-30)
    return -float(np.nanmean(np.gradient(logp, k[m])))


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_psd_analysis(
    vol,
    voxel_nm=0.4,
    *,
    tilt_axis=None,
    band=0.25,
    tail_frac=0.2,
    tol_factor=1.0,
    smooth_w=11,
    k_min_frac=0.10,
    noise_mode="p80",
    mad_sigma=3.0,
    debug=False,
):
    """
    Compute 3D PSD, detect physical axes, and extract 1D profiles.

    The volume is automatically reordered so that after this call:
        out["vol_zyx"]      shape (Z, Y, X)  –  Z is always the tilt axis
        out["axis_1d"]["kz"]                 –  profile along tilt axis
        out["axis_1d"]["kx/ky"]              –  lateral profiles

    Parameters
    ----------
    vol : ndarray, shape (D0, D1, D2)
        Input volume; axes can be in any order.
    voxel_nm : float or (d0, d1, d2)
        Voxel size in nm. Scalar → isotropic.
    tilt_axis : int (0, 1, 2) or None
        Force an array dimension of *vol* as the tilt/Z axis.
        Use this when there is no missing wedge and auto-detection may fail.
        None (default) → the axis with the fastest PSD decay is used.
    band : float
        Half-width of the central averaging slab in nm⁻¹ for 1D profiles.
    tail_frac : float
        Fraction of the profile tail used for noise estimation.
    tol_factor : float
        Noise threshold multiplier (thr = tol_factor × noise).
    smooth_w : int
        Running-median window width.
    k_min_frac : float
        Low-frequency guard (fraction of Nyquist).
    noise_mode : {"p80", "mad"}
        Noise estimator.
    mad_sigma : float
        Sigma multiplier for MAD estimator.
    debug : bool
        Print axis detection summary.

    Returns
    -------
    dict
        vol_zyx      ndarray (Z, Y, X)
        vol_shape    (Z, Y, X)
        centers      {z0, y0, x0}
        voxel_nm     (dz, dy, dx) in physical order
        perm         permutation applied to vol → vol_zyx
        PSD          ndarray (Z, Y, X)  fftshifted 3D PSD
        axes         {fx, fy, fz, dkx, dky, dkz, nyquist}
        planes       {kxky, kxkz, kykz}  central 2D PSD slices
        axis_1d      {kx, ky, kz}  each: {k, raw, smooth, kc, noise, thr, kmin, i0}
        band         band used (nm⁻¹)
        decay_rates  {kx, ky, kz} PSD decay rates on raw vol (for diagnostics)
    """
    vox_raw = (float(voxel_nm),) * 3 if np.isscalar(voxel_nm) else tuple(float(v) for v in voxel_nm)
    D0, D1, D2 = vol.shape

    # ── 3D PSD on raw volume ────────────────────────────────────────────────
    PSD_raw = np.abs(np.fft.fftshift(np.fft.fftn(vol.astype(np.float32)))) ** 2

    # raw frequency axes: dim0↔kz_raw  dim1↔ky_raw  dim2↔kx_raw
    f_raw = [
        np.fft.fftshift(np.fft.fftfreq(D0, vox_raw[0])),   # kz_raw
        np.fft.fftshift(np.fft.fftfreq(D1, vox_raw[1])),   # ky_raw
        np.fft.fftshift(np.fft.fftfreq(D2, vox_raw[2])),   # kx_raw
    ]
    dk_raw = [abs(f[1] - f[0]) for f in f_raw]    # dkz, dky, dkx

    # raw profiles for axis detection
    def _raw_profile(kax):
        prof = _axis_profile(PSD_raw, kax, band, dk_raw[2], dk_raw[1], dk_raw[0])
        freq = f_raw[{"kx": 2, "ky": 1, "kz": 0}[kax]]
        prof = np.asarray(prof, float); freq = np.asarray(freq, float)
        m = np.isfinite(freq) & (freq >= 0) & np.isfinite(prof)
        if m.sum() < 10:
            return {"k": np.array([]), "smooth": np.array([])}
        kp, pp = freq[m], prof[m]
        sm = _moving_median(pp, smooth_w)
        return {"k": kp, "smooth": sm}

    p_raw = {ax: _raw_profile(ax) for ax in ("kx", "ky", "kz")}
    decay = {ax: _decay_rate(p_raw[ax]["k"], p_raw[ax]["smooth"]) for ax in ("kx", "ky", "kz")}

    # ── detect / force tilt axis ────────────────────────────────────────────
    # k-axis name → raw PSD dim index
    k_to_dim = {"kz": 0, "ky": 1, "kx": 2}

    if tilt_axis is None:
        order   = sorted(decay.items(),
                         key=lambda x: x[1] if np.isfinite(x[1]) else -np.inf,
                         reverse=True)
        z_kname = order[0][0]
        xy      = [a for a, _ in order[1:]]
    else:
        if tilt_axis not in (0, 1, 2):
            raise ValueError(f"tilt_axis must be 0, 1, or 2; got {tilt_axis}")
        z_kname = {0: "kz", 1: "ky", 2: "kx"}[tilt_axis]
        remaining = sorted(
            [a for a in ("kx", "ky", "kz") if a != z_kname],
            key=lambda a: -decay.get(a, -np.inf) if np.isfinite(decay.get(a, np.nan)) else -np.inf
        )
        xy = remaining  # higher decay → X, lower → Y

    # permutation: output [Z, Y, X] ← raw dims
    perm = [k_to_dim[z_kname], k_to_dim[xy[1]], k_to_dim[xy[0]]]

    if debug:
        print("\n[AXIS DETECTION]")
        for ax, v in decay.items():
            tag = " ← tilt (Z)" if ax == z_kname else ""
            print(f"  {ax}: decay={v:.3e}{tag}" if np.isfinite(v) else f"  {ax}: nan{tag}")
        print(f"  permutation {perm}: raw dims → (Z, Y, X)")

    # ── reorder vol and PSD to physical (Z, Y, X) ───────────────────────────
    # 3D FFT is separable → transposing vol ↔ transposing its PSD
    vol_zyx = np.transpose(vol, perm)
    PSD     = np.transpose(PSD_raw, perm)

    Z, Y, X = vol_zyx.shape
    z0, y0, x0 = Z // 2, Y // 2, X // 2

    # voxel sizes and freq axes in physical order
    dz = vox_raw[perm[0]]; dy = vox_raw[perm[1]]; dx = vox_raw[perm[2]]
    fz = np.fft.fftshift(np.fft.fftfreq(Z, dz))
    fy = np.fft.fftshift(np.fft.fftfreq(Y, dy))
    fx = np.fft.fftshift(np.fft.fftfreq(X, dx))
    dkz = abs(fz[1] - fz[0]); dky = abs(fy[1] - fy[0]); dkx = abs(fx[1] - fx[0])

    # ── 1D profiles on physically ordered PSD ───────────────────────────────
    def _profile(kax, freq):
        prof = _axis_profile(PSD, kax, band, dkx, dky, dkz)
        prof = np.asarray(prof, float); freq = np.asarray(freq, float)
        m = np.isfinite(freq) & (freq >= 0) & np.isfinite(prof)
        if m.sum() < 10:
            return {"k": np.array([]), "raw": np.array([]), "smooth": np.array([]),
                    "kc": np.nan, "noise": np.nan, "thr": np.nan, "kmin": np.nan, "i0": 0}
        kp, pp = freq[m], prof[m]
        kc, noise, thr, sm, kmin, i0 = _quick_cutoff(
            kp, pp, tail_frac=tail_frac, smooth_w=smooth_w, tol_factor=tol_factor,
            k_min_frac=k_min_frac, noise_mode=noise_mode, mad_sigma=mad_sigma)
        return {"k": kp, "raw": pp, "smooth": sm, "kc": kc,
                "noise": noise, "thr": thr, "kmin": kmin, "i0": i0}

    return {
        "vol_zyx":     vol_zyx,
        "vol_shape":   (Z, Y, X),
        "centers":     {"z0": z0, "y0": y0, "x0": x0},
        "voxel_nm":    (dz, dy, dx),
        "perm":        perm,
        "band":        band,
        "PSD":         PSD,
        "axes": {
            "fx": fx, "fy": fy, "fz": fz,
            "dkx": dkx, "dky": dky, "dkz": dkz,
            "nyquist": (1 / (2 * dx), 1 / (2 * dy), 1 / (2 * dz)),
        },
        "planes": {
            "kxky": PSD[z0, :, :],   # lateral plane  (Y, X)
            "kxkz": PSD[:, y0, :],   # tilt plane XZ  (Z, X)
            "kykz": PSD[:, :, x0],   # tilt plane YZ  (Z, Y)
        },
        "axis_1d": {
            "kx": _profile("kx", fx),   # lateral X
            "ky": _profile("ky", fy),   # lateral Y
            "kz": _profile("kz", fz),   # tilt / Z
        },
        "decay_rates": decay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_real_space(out, cmap="gray", title=""):
    """
    Three orthogonal central slices of the volume in physical nm units.

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    cmap : str or Colormap
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    vol   = out["vol_zyx"]
    dz, dy, dx = out["voxel_nm"]
    Z, Y, X    = out["vol_shape"]
    c          = out["centers"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    kw = dict(cmap=cmap, origin="lower", aspect="equal")
    axes[0].imshow(_norm(vol[c["z0"], :, :]),  extent=[0, X*dx, 0, Y*dy], **kw)
    axes[0].set(title="XY  (Z slice)",  xlabel="x (nm)", ylabel="y (nm)")
    axes[1].imshow(_norm(vol[:, c["y0"], :]),  extent=[0, X*dx, 0, Z*dz], **kw)
    axes[1].set(title="XZ  (Y slice) ← tilt", xlabel="x (nm)", ylabel="z (nm)")
    axes[2].imshow(_norm(vol[:, :, c["x0"]]),  extent=[0, Y*dy, 0, Z*dz], **kw)
    axes[2].set(title="YZ  (X slice) ← tilt", xlabel="y (nm)", ylabel="z (nm)")
    if title:
        fig.suptitle(title, y=1.01)
    plt.tight_layout(); plt.show()
    return fig


def plot_psd_planes(out, *, show_bands=False, vmin_log=None, vmax_log=None,
                   cmap="gray", eps=1e-30, alpha_band=0.25):
    """
    Three principal PSD planes on a log scale, with optional profile-band overlay.

    Planes shown:
        kx–ky  (lateral, no MW)
        kx–kz  (tilt plane, MW visible along kz)
        ky–kz  (tilt plane, MW visible along kz)

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    show_bands : bool
        Overlay the slab regions used to extract 1D profiles. Default False.
    vmin_log, vmax_log : float or None
        Log₁₀ display limits; auto-set if None.
    cmap : str or Colormap
    eps : float
        Floor before log₁₀.
    alpha_band : float
        Transparency of band rectangles when show_bands=True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fx = out["axes"]["fx"]; fy = out["axes"]["fy"]; fz = out["axes"]["fz"]
    dkx = out["axes"]["dkx"]; dky = out["axes"]["dky"]; dkz = out["axes"]["dkz"]
    Z, Y, X  = out["vol_shape"]
    z0, y0, x0 = out["centers"]["z0"], out["centers"]["y0"], out["centers"]["x0"]
    band     = out["band"]

    Ls = [np.log10(np.maximum(out["planes"][k], eps)) for k in ("kxky", "kxkz", "kykz")]
    all_v = np.concatenate([L.ravel() for L in Ls])
    if vmax_log is None: vmax_log = float(np.nanpercentile(all_v, 99.99))
    if vmin_log is None: vmin_log = float(np.nanpercentile(all_v, 10.0))

    kw = dict(cmap=cmap, vmin=vmin_log, vmax=vmax_log, origin="lower", aspect="auto")
    exts   = [[fx[0], fx[-1], fy[0], fy[-1]],
              [fx[0], fx[-1], fz[0], fz[-1]],
              [fy[0], fy[-1], fz[0], fz[-1]]]
    xlbls  = [r"$k_X$ (nm$^{-1}$)", r"$k_X$ (nm$^{-1}$)", r"$k_Y$ (nm$^{-1}$)"]
    ylbls  = [r"$k_Y$ (nm$^{-1}$)", r"$k_Z$ (nm$^{-1}$)", r"$k_Z$ (nm$^{-1}$)"]
    titles = [r"$k_X$–$k_Y$  (lateral)",
              r"$k_X$–$k_Z$  ← MW",
              r"$k_Y$–$k_Z$  ← MW"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, L, ext, xl, yl, t in zip(axes, Ls, exts, xlbls, ylbls, titles):
        im = ax.imshow(L, extent=ext, **kw)
        ax.set(xlabel=xl, ylabel=yl, title=t)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_bands:
        bx = max(1, int(np.round(band / dkx)))
        by = max(1, int(np.round(band / dky)))
        bz = max(1, int(np.round(band / dkz)))

        # slab edges in frequency space
        kx_lo, kx_hi = fx[max(0, x0 - bx)], fx[min(X - 1, x0 + bx)]
        ky_lo, ky_hi = fy[max(0, y0 - by)], fy[min(Y - 1, y0 + by)]
        kz_lo, kz_hi = fz[max(0, z0 - bz)], fz[min(Z - 1, z0 + bz)]

        def _rect(ax_, x0r, y0r, w, h, color, label):
            ax_.add_patch(mpatches.Rectangle(
                (x0r, y0r), w, h, linewidth=1.5,
                edgecolor=color, facecolor=color, alpha=alpha_band, label=label, zorder=3))

        kx_span = fx[-1] - fx[0]; ky_span = fy[-1] - fy[0]; kz_span = fz[-1] - fz[0]

        # kx–ky plane: show kx slab (blue, restricted ky) and ky slab (green, restricted kx)
        _rect(axes[0], fx[0],  ky_lo, kx_span,   ky_hi - ky_lo, "tab:blue",  "kx profile slab")
        _rect(axes[0], kx_lo, fy[0],  kx_hi - kx_lo, ky_span,   "tab:green", "ky profile slab")
        # kx–kz plane
        _rect(axes[1], fx[0],  kz_lo, kx_span,   kz_hi - kz_lo, "tab:blue",  "kx profile slab")
        _rect(axes[1], kx_lo, fz[0],  kx_hi - kx_lo, kz_span,   "tab:red",   "kz profile slab")
        # ky–kz plane
        _rect(axes[2], fy[0],  kz_lo, ky_span,   kz_hi - kz_lo, "tab:green", "ky profile slab")
        _rect(axes[2], ky_lo, fz[0],  ky_hi - ky_lo, kz_span,   "tab:red",   "kz profile slab")

        handles = [
            mpatches.Patch(color="tab:blue",  label="kx slab"),
            mpatches.Patch(color="tab:green", label="ky slab"),
            mpatches.Patch(color="tab:red",   label="kz slab (tilt)"),
        ]
        axes[2].legend(handles=handles, fontsize=7, loc="lower right")

    plt.tight_layout(); plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Lorentzian resolution fitting
# ─────────────────────────────────────────────────────────────────────────────

def lorentz_psd_C(k, A, xi, p, C):
    """
    Lorentzian PSD model with additive pedestal.

    PSD(k) = A / (1 + (2π k ξ)²)^p  +  C

    Parameters
    ----------
    k : array_like
        Spatial frequency (nm⁻¹).
    A : float
        Signal amplitude at k = 0.
    xi : float
        Correlation length (nm).
    p : float
        Decay exponent.
    C : float
        Noise pedestal.
    """
    return A / (1.0 + (2.0 * np.pi * np.asarray(k, float) * xi) ** 2) ** p + C


def fit_lorentz_cutoff(
    k, psd_s,
    *,
    kmin=0.0,
    kmax_frac=0.97,
    min_pts=20,
    noise_tail_frac=0.20,
    noise_kmax_frac=1.0,
    choose_cut="last",
    eps=1e-30,
):
    """
    Fit a Lorentzian+C model to a 1D PSD profile and find the resolution cutoff.

    The cutoff is the crossing of the fitted curve with a linear noise floor
    estimated from the high-k tail (in log space).

    Parameters
    ----------
    k : array_like
        Spatial-frequency axis (nm⁻¹), positive half.
    psd_s : array_like
        Smoothed PSD values (e.g. ``out["axis_1d"]["kz"]["smooth"]``).
    kmin : float
        Lower fit bound (nm⁻¹).
    kmax_frac : float
        Upper fit bound as fraction of max valid frequency.
    min_pts : int
        Minimum points required in fit window.
    noise_tail_frac : float
        Fraction of the profile used to fit the noise line.
    noise_kmax_frac : float
        Trim above this fraction of kmax before noise fit.
    choose_cut : {"first", "last"}
        Which crossing to report when multiple exist.
    eps : float
        Numerical floor before log₁₀.

    Returns
    -------
    dict  always contains ``ok`` (bool)
        On success: A, xi, p, C, their errors, r2_lorentz, fit_range,
                    noise_line, k_cut, res_cut_nm, cut_mode, crossings.
        On failure: reason, fit_range (if established).
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

    C0    = max(float(np.median(y[-max(8, y.size // 6):])), eps)
    A0    = max(float(np.max(y) - C0), float(np.max(y)) * 0.5)
    y0    = np.maximum(y - C0, eps)
    kh    = float(x[np.argmin(np.abs(y0 - np.max(y0) / 2))])
    xi0   = 1.0 / (2 * np.pi * kh + 1e-30) if kh > 0 else 1.0
    C_min = max(float(np.median(y[-max(5, x.size // 10):])), eps)
    C_min = min(C_min, C0 * 0.9)

    def _lm(kv, A, xi, p, C):
        return np.log10(np.maximum(lorentz_psd_C(kv, A, xi, p, C), eps))

    try:
        popt, pcov = curve_fit(
            _lm, x, ylog, p0=[A0, xi0, 1.2, C0],
            bounds=([eps, eps, 0.05, C_min], [np.inf, np.inf, 20, np.inf]),
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
    r2 = 1 - float(np.sum((ylog - yh) ** 2)) / (float(np.sum((ylog - ylog.mean()) ** 2)) + 1e-30)

    # noise line
    bn    = base & (k <= noise_kmax_frac * kmax_data)
    order = np.argsort(k[bn])
    ks, ys = k[bn][order], psd_s[bn][order]
    n_tail = max(8, int(np.ceil(noise_tail_frac * ks.size)))
    kt, pt = ks[-n_tail:], ys[-n_tail:]
    yt = np.log10(np.maximum(pt, eps))
    b_n, a_n = np.polyfit(kt, yt, 1)
    r2n = 1 - float(np.sum((yt - (a_n + b_n * kt)) ** 2)) / (float(np.sum((yt - yt.mean()) ** 2)) + 1e-30)

    # crossings
    ke = np.sort(k[np.isfinite(k) & (k >= kmin)])
    lg = (np.log10(np.maximum(lorentz_psd_C(ke, A, xi, p, C), eps))
          - np.log10(np.maximum(10 ** (a_n + b_n * ke), eps)))

    crosses = []
    for i in range(1, ke.size):
        if np.isfinite(lg[i - 1]) and np.isfinite(lg[i]) and lg[i - 1] * lg[i] <= 0:
            g1, g2, k1, k2 = lg[i - 1], lg[i], ke[i - 1], ke[i]
            crosses.append(float(k2) if g1 == g2 else float(k1 - g1 * (k2 - k1) / (g2 - g1)))

    if crosses:
        kc   = crosses[0] if choose_cut == "first" else crosses[-1]
        mode = "crossing"; gap = 0.0
    else:
        idx  = int(np.argmin(np.abs(lg)))
        kc   = float(ke[idx]); gap = float(lg[idx]); mode = "min_distance"

    return {
        "ok": True,
        "A": A, "xi": xi, "p": p, "C": C,
        "A_err": A_e, "xi_err": xi_e, "p_err": p_e, "C_err": C_e,
        "r2_lorentz": r2,
        "fit_range": (kmin, khi),
        "noise_line": {
            "a": a_n, "b": b_n, "r2": r2n,
            "n_tail": int(kt.size),
            "k_range": (float(kt[0]), float(kt[-1])),
            "k_used": kt.copy(), "psd_used": pt.copy(),
        },
        "k_cut": kc, "res_cut_nm": float(1 / kc) if np.isfinite(kc) and kc > 0 else np.nan,
        "cut_mode": mode, "crossings": crosses, "log_gap_at_cut": gap,
    }


def plot_lorentz_fit(out, axis, fit, title="", save_path=None):
    """
    Diagnostic plot for one Lorentzian resolution fit.

    Parameters
    ----------
    out : dict
        Output of ``compute_psd_analysis``.
    axis : {"kx", "ky", "kz"}
        Which axis profile to plot.
    fit : dict
        Output of ``fit_lorentz_cutoff``.
    title : str
    save_path : str or None
        If given, save the figure at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    d  = out["axis_1d"][axis]
    k  = np.asarray(d["k"], float)
    eps = 1e-30

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(k, np.asarray(d["raw"], float), s=12, alpha=0.4, label="PSD raw")

    if fit.get("ok"):
        A, xi, p, C = fit["A"], fit["xi"], fit["p"], fit["C"]
        lo, hi = fit["fit_range"]
        mf = (k >= lo) & (k <= hi)
        ax.plot(k[mf], lorentz_psd_C(k[mf], A, xi, p, C), "r-", lw=1.8,
                label=f"Lorentz+C  R²={fit['r2_lorentz']:.3f}")
        ax.axhline(C, ls=":", lw=1.2, color="tomato", label=f"C={C:.2e}")

        nl = fit["noise_line"]
        ax.plot(k, 10 ** (nl["a"] + nl["b"] * k), "k--", lw=1.2,
                label=f"Noise  R²={nl['r2']:.2f}")
        ax.scatter(nl["k_used"], nl["psd_used"], s=35, color="k", zorder=5,
                   label=f"noise pts (n={nl['n_tail']})")

        for xc in fit.get("crossings", [])[:-1]:
            ax.axvline(xc, color="gray", ls="--", lw=0.8, alpha=0.5)

        kc = fit["k_cut"]
        if np.isfinite(kc) and kc > 0:
            lbl = f"kc={kc:.3g} nm⁻¹ → {fit['res_cut_nm']:.2f} nm  [{fit['cut_mode']}]"
            ax.axvline(kc, color="purple", ls="-.", lw=2, label=lbl)
        else:
            ax.text(0.5, 0.5, "No cutoff", transform=ax.transAxes, ha="center", color="red")

        xi_e = fit.get("xi_err", np.nan)
        if np.isfinite(xi_e):
            ax.text(0.97, 0.97, f"ξ={xi*1e3:.1f}±{xi_e*1e3:.1f} pm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color="darkred")
    else:
        ax.text(0.5, 0.5, f"Fit failed:\n{fit.get('reason', '')}",
                transform=ax.transAxes, ha="center", color="red")

    ax.set(yscale="log", xlabel=r"Spatial frequency (nm$^{-1}$)",
           ylabel="PSD (a.u.)", title=title or axis)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
