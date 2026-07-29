# DL_etomo — Deep Learning strategies for Electron Tomography

**Recover high-quality and semi-quantitative 3D reconstructions from severely limited-angle, sparse-view, and low-dose electron tomography data** using physics-based, supervised, and unsupervised deep learning approaches.

---

## Overview

Electron tomography under real experimental conditions faces several fundamental challenges:

- **Limited-angle acquisition** — missing wedge artifacts
- **Sparse projections** — undersampled angular coverage
- **Low dose / noisy data** — poor signal-to-noise ratio

Classical reconstruction methods such as SIRT and CS-TV may exhibit severe missing-wedge and noise artifacts in these regimes. This repository provides practical and reproducible solutions based on **Deep Image Prior (DIP)** and its multi-channel extension, **DIPm-TV**, which require no labeled training data.

---

## Methods

### Classical baselines
- SIRT
- CS-TV

### Supervised deep learning
- **U-Net** — supervised restoration of SIRT reconstructions

### Unsupervised deep learning
- **DIP-TV** — single-channel Deep Image Prior with Total Variation regularization
- **DIPm-TV** — multi-channel extension for the joint reconstruction of correlated EDX or EELS volumes

---

## Applications

The methods have been validated on several challenging scenarios:

- **Simulated nanoparticle datasets** — 2D reconstructions across three acquisition regimes:
  - −60° to +60° in 2° increments (62 projections, limited-angle)
  - −60° to +60° in 10° increments (13 projections, sparse-view)
  - −30° to +30° in 2° increments (32 projections, limited-angle with a narrow angular range)
- **Simulated multi-channel phantoms** — EDX and EELS channel phantoms for DIPm-TV validation
- **Experimental 3D tilt-series** — Platinum nanoparticles (-60°:2°:+60°, 62 projections)
- **Experimental STEM-EDX tomography** — Phase-change memory (PCM) devices, Ti/Ge/Sb/Te channels (16 projections, ±40°)
- **Experimental EELS tomography** — Core-loss iron-oxide mapping with Fe²⁺/Fe³⁺ channels (9 projections, ±70°)

---

## Associated Publications

* **Unsupervised Deep Image Prior for Sparse-View and Limited-Angle Electron Tomography**
  S. Brosset, D. del Pozo Bueno, T. David, L. Guetaz, P. Ciuciu, and Z. Saghi
  [Published article in *Ultramicroscopy*](https://www.sciencedirect.com/science/article/pii/S0304399126001063) · [arXiv preprint](https://arxiv.org/abs/2605.27139)

* **Unsupervised Deep Learning for Limited-Angle STEM-EDX Tomography — Application to 3D Chemical Analysis of Phase-Change Memory Devices**
  D. del Pozo Bueno, S. Brosset, T. Monniez, G. Navarro, P. Ciuciu, and Z. Saghi
  [arXiv preprint](https://arxiv.org/abs/2606.10547)

* **Low-Dose 3D Bonding Mapping Through "Soft" Core-Loss EELS Tomography and Unsupervised Deep Learning**
  M. Pelaez-Fernandez, D. del-Pozo-Bueno, A. Teurtrie, S. Brosset, M. Marinova, P. Ciuciu, M. Estrader, G. Salazar-Alvarez, F. Peiró, R. Arenal, S. Estradé, Z. Saghi, and F. De la Peña
  [arXiv preprint](https://arxiv.org/abs/2606.10893)

---

## Installation

### Quick start

```bash
git clone https://github.com/CEA-MetroCarac/DL_etomo.git
cd DL_etomo
```

If you don't already have a working environment with the dependencies below,
create one (optional — skip this if you already have one):

```bash
conda env create -f environment.yml
conda activate dl_etomo
```

Then install `dl_etomo` itself into that environment:

```bash
pip install .
```

> This installs a real copy of `dl_etomo` into your environment's
> `site-packages`. If you plan to actively edit the source and want changes
> picked up without reinstalling, use `pip install -e .` instead.

Open any notebook in `Notebooks/` and run, or use the command-line interface
described below.

> **CUDA note:** `astra-toolbox` is compiled for CUDA 11.8 (conda-forge). The PyTorch `cu118` wheel is backward-compatible with CUDA 12.x drivers (≥ 452.39).

### Requirements

The versions below should remain consistent with `environment.yml` and `pyproject.toml`.

| Package | Version | Install via |
| --- | --- | --- |
| Python | 3.12 | conda |
| PyTorch (CUDA) | ≥ 2.4, CUDA 11.8 build | pip |
| astra-toolbox | 2.2.0 | conda-forge |
| tomosipo | ≥ 0.6.0 | conda-forge |
| numpy | ≥ 2.0 | conda-forge |
| scipy | ≥ 1.15 | conda-forge |
| matplotlib | ≥ 3.10 | conda-forge |
| scikit-image | ≥ 0.25 | conda-forge |
| tifffile | ≥ 2024.1.1 | conda-forge |
| tqdm | ≥ 4.67 | conda-forge |
| jupyterlab | ≥ 4.3 | conda-forge |
| einops | ≥ 0.8 | pip |

---

## Usage

Core source modules are in `dl_etomo/` (an installable package — see
[Installation](#installation)). Notebooks in `Notebooks/Simulated/` and
`Notebooks/Experimental/` provide ready-to-run examples; the same
functionality is also available from the command line (see
[Command-line usage](#command-line-usage) below).

### Notebooks

| Notebook                           | Description                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `simu_dip_reconstruction.ipynb`    | 2D DIP reconstruction from simulated sinogram                     |
| `simu_supervised_restoration.ipynb`| U-Net training and restoration of simulated SIRT reconstructions  |
| `simu_dipm_reconstruction.ipynb`   | Multi-channel DIPm-TV reconstruction (simulated data)             |
| `exp_dip_reconstruction.ipynb`     | 2D DIP reconstruction from experimental data                      |
| `exp_supervised_restoration.ipynb` | U-Net supervised restoration of experimental SIRT reconstructions |
| `EDX_DIPm-TV.ipynb`                | Multi-channel DIPm-TV reconstruction (experimental EDX data)      |
| `EELS_DIPm-TV.ipynb`               | Multi-channel DIPm-TV reconstruction (experimental EELS data)     |
| `CS_TV_reconstruction.ipynb`       | Classical CS-TV (Condat-Vu primal-dual) baseline reconstruction   |
| `Quantification_CL.ipynb`          | Cliff-Lorimer quantification of reconstructed EDX volumes         |
| `psd_resolution_notebook.ipynb`    | Resolution estimation via power spectral density                  |

### Data and pretrained models

- **Datasets**: available in `Data/`
- **Pretrained models**: available in `Trained_models/`

---

## Command-line usage

After `pip install -e .`, every method is also runnable as a script, without
Jupyter:

```bash
python -m dl_etomo <subcommand> [flags]
# or, equivalently, once installed:
dl-etomo <subcommand> [flags]
```

Run `dl-etomo --help` for the full subcommand list, or `dl-etomo
<subcommand> --help` for a given subcommand's flags. Every numeric
hyperparameter flag mirrors the corresponding notebook parameter (e.g.
`--num-iter`, `--lr`, `--noise-reg`, `--lambda-tv`); an optional `--config
run.json` can override any of them from a JSON file, useful for saving and
reproducing a full run configuration.

| Subcommand         | Wraps                                              |
| ------------------- | -------------------------------------------------- |
| `dipm-tv`           | `dipm_tv.run_dipm_tv` — flagship 3D multi-channel DIPm-TV reconstruction |
| `dip-tv`            | `dip.dip_reconstruction` — 2D single-channel DIP    |
| `cs-tv`             | `cs_tv.compress_sensing[_2d]` — classical CS-TV baseline (`cs-tv` extra) |
| `quantify`          | `quantification.quantify_cl_vol` — Cliff-Lorimer EDX quantification |
| `psd-resolution`    | `psd_resolution.compute_psd_analysis` + `fit_lorentz_cutoff` |

`dipm-tv` accepts data in either of two layouts via `--dataset-format
{auto,native,hf}` (auto-detected by default):

- **native** — this repo's own `Data/EDX_data`/`Data/EELS_data` style:
  per-element `*_proj.tif` stacks plus a separate `--angles-file`
  (e.g. `Data/Angles/angles_2.txt`).
- **hf** — the `pfnc-gst-haadf-stem-eds-tomography-b2-d3` HuggingFace
  dataset layout, used unmodified: `derived/edx/elemental_tilt_series/*_stack.tif`
  plus `metadata/angles_deg.txt`, both under the dataset root passed as
  `--input-dir`.

Example, reconstructing this repo's own EDX data headlessly:

```bash
dl-etomo dipm-tv \
    --input-dir Data/EDX_data --dataset-format native --prefix SET \
    --angles-file Data/Angles/angles_2.txt \
    --num-iter 1500 --lr 5e-4 --noise-reg 0.05 --lambda-tv 1e-11 \
    --output-dir Data/EDX_results/SET
```

Same command against a local copy of the HuggingFace dataset:

```bash
dl-etomo dipm-tv \
    --input-dir /path/to/pfnc-gst-haadf-stem-eds-tomography-b2-d3 \
    --output-dir out/pfnc_dipmtv
```

---

## License

The source code in this repository is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [`LICENSE`](LICENSE) for the complete license text.

Datasets and pretrained models may be subject to separate licenses. Consult their corresponding documentation before redistribution or reuse. In particular, the external PFNC GST STEM-EDS dataset is distributed under the [CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Citation

If you use this repository, please cite the following works.

### Deep Image Prior for electron tomography

```bibtex
@article{brosset2026dip_etomo,
  title   = {Unsupervised Deep Image Prior for Sparse-View and Limited-Angle Electron Tomography},
  author  = {Brosset, Serge and del Pozo Bueno, Daniel and David, Thomas and Guetaz, Laure and Ciuciu, Philippe and Saghi, Zineb},
  journal = {Ultramicroscopy},
  volume  = {285},
  pages   = {114414},
  year    = {2026},
  doi     = {10.1016/j.ultramic.2026.114414},
  url     = {https://doi.org/10.1016/j.ultramic.2026.114414}
}
```

### Multi-channel DIPm-TV for STEM-EDX tomography

```bibtex
@misc{delpozobueno2026dipmtv_edx,
  title         = {Unsupervised Deep Learning for Limited-Angle {STEM-EDX} Tomography: Application to 3D Chemical Analysis of Phase-Change Memory Devices},
  author        = {del Pozo Bueno, Daniel and Brosset, Serge and Monniez, Theo and Navarro, Gabriele and Ciuciu, Philippe and Saghi, Zineb},
  year          = {2026},
  eprint        = {2606.10547},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  doi           = {10.48550/arXiv.2606.10547},
  url           = {https://arxiv.org/abs/2606.10547}
}
```
