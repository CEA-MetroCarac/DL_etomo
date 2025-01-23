"""
The Deep Image Prior method is adapted from the original DIP work (D. Ulyanov et al.) :
    https://github.com/DmitryUlyanov/deep-image-prior/tree/master
"""

import sys
sys.path.insert(0, '.')
from model import *
from radon import Radon2D
from utils import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display

import torch
from einops import rearrange

dtype = torch.cuda.FloatTensor

def dip_reconstruction(NUM_ITER, LR, IMG_SIZE, STD_INP_NOISE, NOISE_REG,
                       THETA, INPUT_DEPTH, net,
                       input_sino, degraded_sirt, reference_reco=None,
                       tv_weight=0.0, tv_order=1, SHOW_EVERY=100,
                       given_input=None, state=None, DISPLAY=False, DEVICE='cuda'):
    """
    Perform the reconstruction from a 2D sinogram using the deep image prior approach adapted to tomography.

    Args:
        NUM_ITER: number of DIP iterations
        LR: Learning rate
        IMG_SIZE: output image size (assumed to be square shaped)
        STD_INP_NOISE: Standard-deviation of the input noise
        NOISE_REG: if > 0, value of the perturbation (regularization) applied to the input noise
        THETA: array of angular values (in degrees) corresponding to the sinogram to reconstruct
        INPUT_DEPTH: input noise depth
        net: network
        input_sino: Input sinogram for loss computation
        degraded_sirt: Degraded SIRT reconstruction for visual comparison (displayed during training)
        reference_reco: Reference reconstruction for visual comparison (displayed during training) (optionnal)
        tv_weight: if > 0, apply a TV regularization with this weight (default 0)
        tv_order: TV order (default 1)
        SHOW_EVERY: if DISPLAY==True, Display live update every SHOW_EVERY iterations
        given_input: custom input (optionnal), if None is given, uniform noise is used
        state: pretrained model dictionary (weigths & optimizer state) (default None)
        DISPLAY: for jupyter notebook use, if True, display live per iteration update (default False)

    Returns:
        dictionary: 
            best_loss: Best loss reached during training
            best_output: reconstruction corresponding to the best loss
            best_i: iteration number of the best loss
            loss_values: list of loss values
            net: Trained network
            out_avg: Average output after the complete optimization process (using Exponentioal Moving Average, EMA)
            best_input: Regularized input which resulted in the best output (with minimal loss)
            list_iter_reco: List of generated reconstructions per iterations
            training_state: network & optimizer state to resume optimization if needed
    """

    # Specify an input noise, and use an uniform distribution otherwise
    if given_input==None:
        net_input = torch.zeros([1, INPUT_DEPTH, IMG_SIZE, IMG_SIZE])
        net_input = (net_input.uniform_() * STD_INP_NOISE).type(dtype)
        net_input_orig = net_input.clone()
    else:
        net_input = given_input.clone()
        net_input_orig = net_input.clone()

    # Load a pretrained network
    if state != None:
        net.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

    net = net.to(DEVICE)
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=LR)
    loss = torch.nn.MSELoss(reduction='sum').type(dtype)

    radon_op = Radon2D(size=IMG_SIZE, angle=np.deg2rad(np.flip(THETA)))
    radon_op_full = Radon2D(size=IMG_SIZE, angle=np.deg2rad(np.arange(0.,180.,1.)))
    grad_operator = get_torch_grad_op((IMG_SIZE, IMG_SIZE), tv_order).to(DEVICE)

    loss_values = []
    best_loss = 1e9
    best_output = None
    best_i = 0
    out_avg = None
    list_iter_reco = []

    if DISPLAY:
        dh = None

    for it in tqdm(range(NUM_ITER)):

        for param in net.parameters():
            param.grad = None

        # Noise based regularization of the input noise
        if NOISE_REG > 0:
            net_input = net_input_orig + (torch.zeros(net_input.shape).to(DEVICE).normal_() * NOISE_REG)

        out = net(net_input)
        out = rearrange(out, '1 1 h w -> h w')

        out_sino = radon_op.forward(out)
        out_sino = rearrange(out_sino, 'h w -> 1 1 h w')

        # Computation of the loss on the sinogram of the generated reconstruction
        total_loss = loss(out_sino, input_sino)

        # Possible use of a TV regularization on the generated reconstruction
        if tv_weight > 0:
            total_loss += tv_weight * compute_sparse_tv(out, grad_operator)

        total_loss.backward()

        # Save per iteration reconstruction and loss
        loss_values.append(total_loss.item())
        list_iter_reco.append(simplify(out))

        # For jupyter notebooks : possible live display of the progress of the reconstruction 
        if DISPLAY and ((it+1)%SHOW_EVERY==0 or (it+1)==1):
            dh = plot_function(dh, [loss_values, 
                                simplify(radon_op_full.forward(out)),
                                sinoToFullView(input_sino, np.flip(THETA)), 
                                simplify(out), 
                                simplify(degraded_sirt),
                                reference_reco])

        loss_value = total_loss.item()                
        if loss_value < best_loss:
            best_loss = loss_value
            best_i = it
            best_output = simplify(net(net_input))
            best_input = simplify(net_input)
            
        optimizer.step()

    return {
        'best_loss':best_loss, 
        'best_output':best_output,
        'best_i':best_i, 
        'loss_values':loss_values, 
        'net':net, 
        'out_avg':simplify(out_avg), 
        'best_input':best_input, 
        'list_iter_reco':list_iter_reco, 
        'training_state': {'model_state_dict':net.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}
    }


def plot_function(dh, data):
    """
    Plot function for jupyter notebooks to display a live evolution of : the loss and the generated reconstruction and sinogram compared to a reference
    """

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax[0][0].set_yscale('log')
    ax[0][0].plot(data[0], color='blue')
    ax[0][0].set_yscale('log')
    ax[0][0].set_title('Training loss')

    ax[0][1].imshow(data[1], cmap='gray', aspect='auto', extent=[0, data[3].shape[1], 90, -90])
    ax[0][1].set_title('Generated DIP sinogram')
    ax[0][2].imshow(data[2], cmap='gray', aspect='auto', extent=[0, data[3].shape[1], 90, -90])
    ax[0][2].set_title('Reference sinogram (For loss computation)')

    if data[5] is None:
        ax[1][0].set_visible(False)
    else:
        ax[1][0].imshow(simplify(data[5]), cmap='gray')
        ax[1][0].set_title('SIRT reference')

    ax[1][1].imshow(data[3], cmap='gray')
    ax[1][1].set_title('Generated DIP reconstruction')
    ax[1][1].axis('off')
    ax[1][2].imshow(data[4], cmap='gray')
    ax[1][2].set_title('Degraded SIRT reconstruction')
    ax[1][2].axis('off')

    if dh is None:
        dh = display.display(fig, display_id=True)
    else:
        display.update_display(fig, display_id=dh.display_id)

    plt.close(fig)

    return dh