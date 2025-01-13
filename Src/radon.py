"""
Create a Radon operator class based on Tomosipo Library
The operator is used to :
    Generate a sinogram from a 2D or 3D image
    Compute the reconstruction using FB or SIRT method

The SIRT reconstruction algorithm is taken from Tomosipo official implementation (A. Hendriksen et al.) :
    https://github.com/ahendriksen/tomosipo
"""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

import tomosipo as ts
from tomosipo.torch_support import (
    to_autograd,
)

class Radon2D(nn.Module):
    def __init__(self, size=256, angle=180, device='cuda'):
        """2D Radon transform with forward and backword operator

        Args:
            size (int, optional): image size. Defaults to 256.
            angle (int, optional): array of angle, in radian. Defaults to 180.
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
        """
        super().__init__()
        self.img_size = size
        self.angle = angle
        self.device = device

        self.init_op()
        
    def init_op(self):
        self.vg = ts.volume(size=1, shape=(1, self.img_size, self.img_size))
        self.pg = ts.parallel(angles=self.angle, shape=(1, self.img_size), size=(1, 1))
        self.operator = ts.operator(self.vg, self.pg)
    
    def forward(self, img):
        sino = to_autograd(self.operator, is_2d=True)(img)
        
        return sino

    def bp(self, sino):
        reco = to_autograd(self.operator.T, is_2d=True)(sino)

        return reco

    def backward_sirt_ts(self, sino, progress_bar=False, min_constraint=True, num_iters=100):
        """
        SIRT reconstruction from a 2D sinogram

        Args:
            sino (torch tensor): input 2D sinogram of shape [number of projections, image size]
            progress_bar (bool, optional): display recosntruction progress bar. Defaults to False.
            min_constraint (bool, optional): Force positiv values in the reconstruction. Defaults to True.
            num_iters (int, optional): number of iterations. Defaults to 100.

        Returns:
            2D reconstructions of shape [image size, image size]
        """
         
        y = sino.to(self.device)
        x_cur = torch.zeros(self.operator.domain_shape, device=self.device, requires_grad=True)

        C = to_autograd(self.operator.T)(torch.ones(self.operator.range_shape, device=self.device, requires_grad=True))
        C[C < ts.epsilon] = np.Inf
        C.reciprocal_()

        R = to_autograd(self.operator)(torch.ones(self.operator.domain_shape, device=self.device,  requires_grad=True))
        R[R < ts.epsilon] = np.Inf
        R.reciprocal_()

        if progress_bar:
            for i in tqdm(range(num_iters)):
                x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
                x_cur = torch.clone(x_new)    
                if min_constraint:
                    x_cur[x_cur<0] = 0
        else :
            for i in range(num_iters):
                x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
                x_cur = torch.clone(x_new)    
                if min_constraint:
                    x_cur[x_cur<0] = 0
        return x_cur
    

class Radon3D(nn.Module):
        """3D Radon transform with forward and backword operator

        Args:
            size (int, optional): image size. Defaults to 256.
            angle (int, optional): array of angle, in radian. Defaults to 180.
            depth (int): depth of the 3D volume
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
        """
    def __init__(self, depth, size=256, angle=np.arange(0.,180.,1.), device='cuda'):
        super().__init__()
        self.img_size = size
        self.angle = angle
        self.depth = depth
        self.device = device
        
        self.init_op()
        
    def init_op(self):
        self.vg = ts.volume(size=1, shape=(self.depth, self.img_size, self.img_size))
        self.pg = ts.parallel(angles=self.angle, shape=(self.depth, self.img_size), size=(1, 1))
        self.operator = ts.operator(self.vg, self.pg)
    
    def forward(self, img):
        sino = to_autograd(self.operator)(img)
        
        return sino

    def bp(self, sino):
        reco = to_autograd(self.operator.T)(sino)

        return reco

    def backward_sirt_ts(self, sino, progress_bar=False, min_constraint=True, num_iters=100):
        """
        SIRT reconstruction from a 3D sinogram

        Args:
            sino (torch tensor): input 3D sinogram of shape [image depth (3D), number of projections, image size]
            progress_bar (bool, optional): display recosntruction progress bar. Defaults to False.
            min_constraint (bool, optional): Force positiv values in the reconstruction. Defaults to True.
            num_iters (int, optional): number of iterations. Defaults to 100.

        Returns:
            3D reconstructions of shape [image depth, image size, image size]
        """

        y = sino.to(self.device)
        x_cur = torch.zeros(self.operator.domain_shape, device=self.device, requires_grad=True)

        C = to_autograd(self.operator.T)(torch.ones(self.operator.range_shape, device=self.device, requires_grad=True))
        C[C < ts.epsilon] = np.Inf
        C.reciprocal_()

        R = to_autograd(self.operator)(torch.ones(self.operator.domain_shape, device=self.device,  requires_grad=True))
        R[R < ts.epsilon] = np.Inf
        R.reciprocal_()

        if progress_bar:
            for i in tqdm(range(num_iters)):
                x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
                x_cur = torch.clone(x_new)
                if min_constraint:
                    x_cur[x_cur<0] = 0
        else:
            for i in range(num_iters):
                x_new = x_cur + C * self.operator.T(R * (y - self.operator(x_cur)))
                x_cur = torch.clone(x_new)
                if min_constraint:
                    x_cur[x_cur<0] = 0

        return x_cur
