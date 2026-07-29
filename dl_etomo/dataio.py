"""
dataio.py
=========
Load a multi-channel tilt series (per-element/phase projection stacks +
tilt angles) from either of the two directory layouts this project needs
to support unmodified:

``native``
    This repo's own layout, e.g. ``Data/EDX_data/``: one TIFF stack per
    element named ``{prefix}_{element}_proj.tif`` (or just
    ``{element}_proj.tif`` when there is no sample prefix, as in
    ``Data/EELS_data/``), each of shape ``(N_angles, H, W)``, with tilt
    angles in a separate ``angles_*.txt`` file (one value per line).

``hf``
    The layout of the ``pfnc-gst-haadf-stem-eds-tomography-b2-d3``
    HuggingFace dataset: ``derived/edx/elemental_tilt_series/{element}_stack.tif``
    plus ``metadata/angles_deg.txt``, both rooted under the same
    ``input_dir``.

Public API
----------
load_tilt_series(input_dir, format='auto', prefix=None, angles_file=None,
                 phase_names=None)
    -> (sinograms, angles_deg, phase_names)
"""

from pathlib import Path

import numpy as np
import tifffile as tiff

_NATIVE_SUFFIX = "_proj.tif"
_HF_SUFFIX = "_stack.tif"
_HF_TILT_SUBDIR = Path("derived") / "edx" / "elemental_tilt_series"
_HF_ANGLES_FILE = Path("metadata") / "angles_deg.txt"


def _detect_format(input_dir):
    input_dir = Path(input_dir)
    if list((input_dir / _HF_TILT_SUBDIR).glob(f"*{_HF_SUFFIX}")):
        return "hf"
    if list(input_dir.glob(f"*{_NATIVE_SUFFIX}")):
        return "native"
    raise ValueError(
        f"Could not auto-detect a known dataset layout under '{input_dir}'. "
        f"Expected either '*{_NATIVE_SUFFIX}' files directly inside it (native), "
        f"or a '{_HF_TILT_SUBDIR}/*{_HF_SUFFIX}' subtree (hf). "
        f"Pass format='native' or format='hf' explicitly to skip detection."
    )


def _discover_native(input_dir, prefix):
    input_dir = Path(input_dir)
    pattern = f"{prefix}_*{_NATIVE_SUFFIX}" if prefix else f"*{_NATIVE_SUFFIX}"
    files = {}
    for path in sorted(input_dir.glob(pattern)):
        stem = path.name[: -len(_NATIVE_SUFFIX)]
        if prefix:
            stem = stem[len(prefix) + 1:]
        files[stem] = path
    return files


def _discover_hf(input_dir, prefix):
    tilt_dir = Path(input_dir) / _HF_TILT_SUBDIR
    files = {}
    for path in sorted(tilt_dir.glob(f"*{_HF_SUFFIX}")):
        stem = path.name[: -len(_HF_SUFFIX)]
        files[stem] = path
    if prefix:
        files = {k: v for k, v in files.items() if k == prefix}
    return files


def load_tilt_series(input_dir, format="auto", prefix=None, angles_file=None,
                      phase_names=None):
    """
    Load a stacked multi-channel tilt series ready for
    ``dl_etomo.dipm_tv.preprocess_sinograms``.

    Parameters
    ----------
    input_dir : str or Path
        Directory holding the projection stacks (native), or the dataset
        root containing ``derived/`` and ``metadata/`` (hf).
    format : {'auto', 'native', 'hf'}
        Dataset layout. 'auto' picks 'hf' if a
        ``derived/edx/elemental_tilt_series`` subtree with ``*_stack.tif``
        files exists under ``input_dir``, else 'native' if ``*_proj.tif``
        files sit directly inside ``input_dir``.
    prefix : str or None
        Sample prefix to filter on when a folder mixes several samples,
        e.g. 'SET' or 'Virgin' in ``Data/EDX_data`` (native), or an element
        name to keep just one channel (hf). None keeps everything found.
    angles_file : str or Path or None
        Path to a whitespace/newline-separated file of tilt angles in
        degrees. Required for 'native' (there is no fixed convention
        linking a projection folder to its angles file). Ignored for 'hf',
        which always reads ``metadata/angles_deg.txt`` under ``input_dir``.
    phase_names : list of str or None
        Explicit channel order/selection (e.g. ``['Ge', 'Sb', 'Te']``).
        None keeps every discovered channel, sorted alphabetically.

    Returns
    -------
    sinograms : np.ndarray, shape (N_ch, N_angles, H, W)
    angles_deg : np.ndarray, shape (N_angles,)
    phase_names : list of str
        Channel names in the same order as ``sinograms``' first axis.
    """
    input_dir = Path(input_dir)
    if format == "auto":
        format = _detect_format(input_dir)

    if format == "native":
        files = _discover_native(input_dir, prefix)
        if angles_file is None:
            raise ValueError(
                "angles_file is required for format='native' -- there is no "
                "naming convention linking a projection folder to its angles "
                "file (e.g. pass Data/Angles/angles_2.txt)."
            )
        angles_deg = np.loadtxt(angles_file, dtype=np.float64).reshape(-1)
    elif format == "hf":
        files = _discover_hf(input_dir, prefix)
        angles_deg = np.loadtxt(input_dir / _HF_ANGLES_FILE, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"Unknown format '{format}', expected 'auto', 'native' or 'hf'.")

    if not files:
        raise FileNotFoundError(
            f"No projection stacks found under '{input_dir}' for format='{format}'"
            + (f" with prefix='{prefix}'." if prefix else ".")
        )

    if phase_names is None:
        phase_names = sorted(files)
    else:
        missing = [name for name in phase_names if name not in files]
        if missing:
            raise KeyError(
                f"Requested phase_names {missing} not found under '{input_dir}'. "
                f"Available: {sorted(files)}"
            )

    stacks = [tiff.imread(str(files[name])) for name in phase_names]
    shapes = {stack.shape for stack in stacks}
    if len(shapes) > 1:
        raise ValueError(
            f"Projection stacks have mismatched shapes across channels: "
            f"{dict(zip(phase_names, (s.shape for s in stacks)))}"
        )

    sinograms = np.stack(stacks, axis=0).astype(np.float32)

    if sinograms.shape[1] != angles_deg.shape[0]:
        raise ValueError(
            f"Number of projections in the data ({sinograms.shape[1]}) does not "
            f"match the number of tilt angles ({angles_deg.shape[0]})."
        )

    return sinograms, angles_deg, phase_names
