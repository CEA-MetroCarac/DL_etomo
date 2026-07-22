# DL_etomo — Deep Learning strategies for Electron Tomography

**Recover high-quality and semi-quantitative 3D reconstructions from severely limited-angle, sparse-view, and low-dose electron tomography data** using physics-based, supervised, and unsupervised deep learning approaches.

---

## Overview

Electron tomography in real experimental conditions faces fundamental challenges:

- **Limited-angle acquisition** — missing wedge artifacts
- **Sparse projections** — undersampled angular coverage
- **Low dose / noisy data** — poor signal-to-noise ratio

Classical methods (SIRT, CS-TV) struggle in these regimes. This repository provides practical, reproducible solutions based on **Deep Image Prior (DIP)** and its multi-channel extension (**DIPm-TV**), requiring no labelled training data.

---

## Methods

### Classical baselines
- SIRT
- CS-TV

### Supervised deep learning
- **U-Net** restoration of SIRT reconstructions

### Unsupervised deep learning
- **DIP-TV** — Deep Image Prior with Total Variation regularization
- **DIPm-TV** — Multi-channel extension for joint reconstruction of correlated volumes (EDX / EELS channels)

---

## Applications

Validated on multiple challenging scenarios:

- **Simulated nanoparticle datasets** — 2D reconstructions across three acquisition regimes:
  - -60°:2°:+60° (63 projections, limited-angle)
  - -60°:10°:+60° (13 projections, sparse)
  - -30°:2°:+30° (31 projections, limited-angle + narrow range)
- **Simulated multi-channel phantoms** — EDX and EELS channel phantoms for DIPm-TV validation
- **Experimental 3D tilt-series** — Platinum nanoparticles (-60°:2°:+60°, 63 projections)
- **Experimental STEM-EDX tomography** — Phase-change memory (PCM) devices, Ti/Ge/Sb/Te channels (16 projections, ±40°)
- **Experimental EELS tomography** — Core-loss iron-oxide mapping, Fe²⁺/Fe³⁺ channels (9 projections, ±70°)

---

## Associated Publications

- **Unsupervised Deep Image Prior for Sparse-View and Limited-Angle Electron Tomography**  
  S. Brosset, D. del Pozo Bueno, T. David, L. Guetaz, P. Ciuciu, Z. Saghi  
  https://arxiv.org/abs/2605.27139

- **Unsupervised Deep Learning for Limited-Angle STEM-EDX Tomography — Application to 3D Chemical Analysis of Phase-Change Memory Devices**  
  D. del Pozo Bueno, S. Brosset, T. Monniez, G. Navarro, P. Ciuciu, Z. Saghi  
  https://arxiv.org/abs/2606.10547

- **Low-Dose 3D Bonding Mapping Through "Soft" Core-Loss EELS Tomography and Unsupervised Deep Learning**  
  M. Pelaez-Fernandez, D. del Pozo Bueno, A. Teurtrie, S. Brosset, M. Marinova, P. Ciuciu, M. Estrader, G. Salazar-Alvarez, F. Peiró, R. Arenal, S. Estradé, Z. Saghi, F. De la Peña  
  https://arxiv.org/abs/2606.10893

---

## Installation

### Quick start

```bash
git clone https://github.com/CEA-MetroCarac/DL_etomo.git
cd DL_etomo
conda env create -f environment.yml
conda activate dl_etomo
```

Then install `dl_etomo` itself as an editable package into that environment:

```bash
pip install -e .
# or, to also pull the classical CS-TV baseline's heavier dependencies:
pip install -e ".[cs-tv]"
```

Open any notebook in `Notebooks/` and run, or use the command-line interface
described below.

> **CUDA note:** `astra-toolbox` is compiled for CUDA 11.8 (conda-forge). The PyTorch `cu118` wheel is backward-compatible with CUDA 12.x drivers (≥ 452.39).

### Requirements

| Package        | Version      | Install via         |
| -------------- | ------------ | ------------------- |
| Python         | ≥ 3.12       | conda               |
| PyTorch (CUDA) | ≥ 2.4        | pip (cu118 wheel)   |
| astra-toolbox  | = 2.2.0      | conda-forge         |
| tomosipo       | ≥ 0.6.0      | conda-forge         |
| numpy          | ≥ 2.0        | conda-forge         |
| scipy          | ≥ 1.15       | conda-forge         |
| matplotlib     | ≥ 3.10       | conda-forge         |
| scikit-image   | ≥ 0.25       | conda-forge         |
| tifffile       | ≥ 2024.1.1   | conda-forge         |
| tqdm           | ≥ 4.67       | conda-forge         |
| jupyterlab     | ≥ 4.3        | conda-forge         |
| einops         | ≥ 0.8        | pip                 |

> **NumPy 2.x required.** The codebase uses `np.inf` (lowercase), which replaced the removed `np.Inf` alias in NumPy 2.0.

---

## Usage

Core source modules are in `dl_etomo/` (an installable package — see
[Installation](#installation)). Notebooks in `Notebooks/Simulated/` and
`Notebooks/Experimental/` provide ready-to-run examples; the same
functionality is also available from the command line (see
[Command-line usage](#command-line-usage) below).

### Source modules

| File                 | Description                                                          |
| -------------------- | --------------------------------------------------------------------- |
| `dip.py`             | 2D DIP training loop with optional live notebook visualization        |
| `dipm_tv.py`         | 3D multi-channel DIPm-TV: CNN3D architecture, TV loss, training loop   |
| `cs_tv.py`           | Classical CS-TV baseline (Condat-Vu primal-dual, `cs-tv` extra)        |
| `model.py`           | 2D U-Net for supervised restoration                                   |
| `radon.py`           | 2D/3D Radon forward/backprojection and SIRT operators (Tomosipo)      |
| `utils.py`           | Normalization, sinogram utilities, MS-SSIM loss                       |
| `psd_resolution.py`  | 3D PSD computation, Lorentzian fitting, resolution estimation         |
| `quantification.py`  | Cliff-Lorimer EDX quantification                                     |
| `kfactors_db.py`     | K-factor / atomic weight reference tables used by `quantification.py` |
| `dataio.py`          | Tilt-series loader supporting both this repo's own data layout and the `pfnc-gst-haadf-stem-eds-tomography` HuggingFace dataset layout |
| `cli/`               | Argparse command-line entry points, one module per subcommand         |

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
| `psd_resolution_notebook.ipynb`    | Resolution estimation via power spectral density                  |

### Data and pretrained models

- **Datasets**: will be released via Zenodo
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

## Key Features

- End-to-end pipeline: reconstruction and restoration
- Unsupervised learning — no ground truth required
- Multi-channel tomography support (EDX / EELS)
- Designed for low-dose, sparse, and limited-angle data
- Fully reproducible via Jupyter notebooks

---

## Roadmap

- [ ] Zenodo dataset release
- [ ] Benchmarks against recent DL methods

---

## License

GPL-3.0. See [LICENSE](LICENSE).

## Citation

If you use this repository, please cite the associated papers listed above.
