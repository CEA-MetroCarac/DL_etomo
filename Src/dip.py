"""
The Deep Image Prior method is taken and adapted from the original DIP work (Ulyanov et al.) :
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
                       THETA, INPUT_DEPTH, 
                       input_sino, cmp_reco, net, 
                       tv_weight=0.0, tv_order=1, SHOW_EVERY=100, 
                       given_input=None, state=None, DISPLAY=True, DEVICE='cuda'):
    """
    Perform DIP reconstruction from a 2D sinogram

    Args:
        NUM_ITER: number of DIP iterations
        LR: Learning rate
        IMG_SIZE: output image size (square shape)
        STD_INP_NOISE: input noise std
        NOISE_REG: input noise perturbation per iter
        THETA : np.arange of angle values (°)
        INPUT_DEPTH: input noise depth
        input_sino: Input comparison sinogram
        cmp_reco : reco to compare with output (displayed during training)
        net : network
        tv_weight: TV regularization weight (default None)
        tv_order : TV order (default 1)
        SHOW_EVERY: display while train every _ iters
        given_input: give a specific input (default None)
        state: pretrained model dict (weigth & optim state) (defautl None)
        DISPLAY: display per iter resutls (default True)

    Returns:
        dictionary: 
            best loss
            best output
            best iteration
            list of losses values
            trained network
            used input
            last avg out (EMA)
            best input (from regularisation)
            list of reco per iteration
            model state (weights & optimizer)
    """

    if given_input==None:
        net_input = torch.zeros([1, INPUT_DEPTH, IMG_SIZE, IMG_SIZE])
        net_input = (net_input.uniform_() * STD_INP_NOISE).type(dtype)
        net_input_orig = net_input.clone()
    else:
        net_input = given_input.clone()
        net_input_orig = net_input.clone()

    if state != None:
        net.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

    net = net.to(DEVICE)
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=LR)
    loss = torch.nn.MSELoss(reduction='sum').type(dtype)

    radon_op = Radon2D(size=IMG_SIZE, angle=np.deg2rad(np.flip(THETA)))
    grad_operator = get_torch_grad_op((IMG_SIZE, IMG_SIZE), tv_order).to(DEVICE)

    loss_values = []
    best_loss = 1e9
    best_output = None
    best_i = 0
    out_avg = None
    list_iter_reco = []

    if DISPLAY:
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        fig.delaxes(ax[1][0])
        dh = display.display(fig, display_id=True)
        ax[0][0].set_yscale('log')

    for it in tqdm(range(NUM_ITER)):

        for param in net.parameters():
            param.grad = None

        # Noise based regularization
        if NOISE_REG > 0:
            net_input = net_input_orig + (torch.zeros(net_input.shape).to(DEVICE).normal_() * NOISE_REG)

        out = net(net_input)
        out = rearrange(out, '1 1 h w -> h w')

        out_sino = radon_op.forward(out)
        out_sino = rearrange(out_sino, 'h w -> 1 1 h w')

        total_loss = loss(out_sino, input_sino)
        if tv_weight > 0:
            total_loss += tv_weight * compute_sparse_tv(out, grad_operator)

        total_loss.backward()

        # Save every iteration
        loss_values.append(total_loss.item())
        list_iter_reco.append(simplify(out))

        if DISPLAY:
            if (it+1)%SHOW_EVERY==0:
                ax[0][0].cla() 
                ax[0][0].plot(loss_values, color='blue')
                ax[0][0].set_yscale('log')
                ax[0][0].set_title('Training loss')

                ax[0][1].cla() 
                ax[0][1].imshow(simplify(out_sino), cmap='gray', aspect='auto')
                ax[0][1].set_title('DIP output sinogram')
                ax[0][1].axis('off')
                ax[0][2].cla() 
                ax[0][2].imshow(simplify(input_sino), cmap='gray', aspect='auto')
                ax[0][2].set_title('Reference sinogram')
                ax[0][2].axis('off')

                ax[1][1].cla() 
                ax[1][1].imshow(simplify(net(net_input)), cmap='gray')
                ax[1][1].set_title('DIP reconstruction')
                ax[1][1].axis('off')
                ax[1][2].cla() 
                ax[1][2].imshow(simplify(cmp_reco), cmap='gray')
                ax[1][2].set_title('Reference')
                ax[1][2].axis('off')
                dh.update(fig)
                
        loss_value = total_loss.item()                
        if loss_value < best_loss:
            best_loss = loss_value
            best_i = it
            best_output = simplify(net(net_input))
            best_input = simplify(net_input)

        optimizer.step()
    if DISPLAY:
        plt.close(fig)

    return {
        'best_loss':best_loss, 
        'best_output':best_output,
        'best_i':best_i, 
        'loss_values':loss_values, 
        'net':net, 
        'used_net_input':net_input,
        'out_avg':simplify(out_avg), 
        'best_input':best_input, 
        'list_iter_reco':list_iter_reco, 
        'training_state': {'model_state_dict':net.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}
    }



def Plot_function():
    """
    Plotting function to display a live update of the reconstruction in jupyter notebooks
    """

    return