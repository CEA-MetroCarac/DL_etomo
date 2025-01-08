This repository contains python functions and jupyter notebooks to perform electron tomography reconstruction and restoration from nanoparticles datasets, as presented in the paper **[]**.

# Electron tomography reconstruction using supervised and unsupervised approaches

The paper compare classical approaches (SIRT, CS-TV), supervised approach (U-Net restoration) and unsupervised approach (Deep Image Prior) for the recosntruction and restoration of electron tomgoraphy data. The data sets consist of both experimental Pt nanoparticles acquired in an TEM and simulated data with similar features.


Description of the paper (ET reco, comparison, rea data) and data (simu & real data sets described from the article) and methods (sirt, tv regularization, supervised restoration, dip reconstruction)
Show results examples from the paper

### Dependencies

The notebooks were tested with the following packages and versions : 

```bash
- Python = 3.12.7
- Tomosipo = 0.6.0
- Numpy = 1.26.4
- tqdm = 4.66.5
- Matplotlib = 3.9.2
- IPython = 8.29.0
- Pytorch = 2.4.1
- Einops = 0.8.0
- Scikit-image = 0.24.0
- Scipy = 1.14.1
- Pytorch_msssim = 1.0.0
```

## How to use

The source code is presented in ```Src``` directory.
Several notebooks are presented in ```Notebooks``` :
- ```/Simulated``` contains supervised U-Net training and DIP reconstruction on the simulated data
- ```/Experimental``` contains the application of the trained network for real data restoration and the DIP reconstruction
- ```Compare_results.ipynb``` compare the results in a similar way to the results of the paper.

###

Data and/or model weights will be available from Zenodo (?) (right now in Data folder)