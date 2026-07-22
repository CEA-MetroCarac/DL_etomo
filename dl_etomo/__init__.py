"""
dl_etomo
========
Deep learning strategies for electron tomography: Deep Image Prior (DIP),
its multi-channel TV-regularized extension (DIPm-TV), classical CS-TV,
Cliff-Lorimer EDX quantification, and PSD-based resolution estimation.

Submodules are imported directly, e.g.:

    from dl_etomo.dipm_tv import CNN3D, run_dipm_tv, preprocess_sinograms

This keeps ``import dl_etomo`` itself cheap (no eager torch import).
"""

__version__ = "0.1.0"
