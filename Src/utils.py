import numpy as np
import torch
from torch import nn
import scipy
from pytorch_msssim import  ms_ssim

def simplify(x):
    """
    for a pytorch tensor : remove empty dimensions, detach and switch to numpy
    for a numpy array : remove empty dimensions
    """
    if torch.is_tensor(x):
        return torch.squeeze(x).detach().cpu().numpy()
    else:
        return np.squeeze(x)

def normalize(x):
    """
    Normalize values of the np array or torch tensor between [0-1]
    """
    if torch.is_tensor(x):
        return (x - torch.amin(x)) / (torch.amax(x) - torch.amin(x))
    else :
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    
def to8bit(array):
    """
    Array to 8 bit format
    """
    array = simplify(array)
    array = (normalize(array)*255).astype(np.uint8)
    return array

def sinoToFullView(sinogram, angle):
    """Return a sinogram of size 180, with missing projections visible as empty spaces
    Args:
        sinogram: sinogram 2D with missing projections, of shape (nb_projections x img_size)
        angle: array of corresponding angles (°)
    """

    sinogram = simplify(sinogram)
    size = sinogram.shape[1]
    full = np.zeros((180, size))

    for idx, theta in enumerate(angle):
        full[theta, :] = sinogram[idx, :]

    return full    

def get_torch_grad_op(img_shape, order):
    """
    From pysap-etomo : https://github.com/CEA-COSMIC/pysap-etomo

    Construct the sparse gradient operator
    """

    img_size = img_shape[0]
    filt = np.zeros((order + 1, 1))
    for k in range(order + 1):
        filt[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

    offsets_x = np.arange(order + 1)
    offsets_y = img_size * np.arange(order + 1)
    shape = (img_size ** 2,) * 2
    sparse_mat_x = scipy.sparse.diags(filt,
                                      offsets=offsets_x, shape=shape).astype(np.float32)
    sparse_mat_y = scipy.sparse.diags(filt,
                                      offsets=offsets_y, shape=shape).astype(np.float32)
    op_matrix = scipy.sparse.vstack([sparse_mat_x, sparse_mat_y])

    indices = torch.from_numpy(np.vstack((op_matrix.row, op_matrix.col)))
    values = torch.from_numpy(op_matrix.data)
    size = torch.Size(op_matrix.shape)
    pytorch_op = torch.sparse_coo_tensor(indices, values, size)

    return pytorch_op

def compute_sparse_tv(img, op):
    """
    Compute the Total Variation from a 2D square image using the sparse gradient operator obtained from the function get_torch_grad_op()
    """
    grad_torch = torch.sparse.mm(op, img.reshape((img.shape[0]*img.shape[0],1)))
    grad_torch = grad_torch.reshape(2*img.shape[0], img.shape[0])
    
    tv = torch.sum(torch.pow(torch.pow(grad_torch[:img.shape[0],:], 2) + torch.pow(grad_torch[img.shape[0]:,:], 2) + 1e-9, 0.5))
    
    return tv

def l1_mssim_loss(gt, pred, alpha=0.5):
    """
    Compute the mixed L1 - MSSSIM loss as presented in : H. Zhao et al. 
        Loss functions for image restoration with neural networks.
    """

    msssim_loss = 1 - ms_ssim(torch.abs(gt), torch.abs(pred), data_range=1, size_average=True )
    l1 = nn.L1Loss()
    l1_loss = l1(gt, pred)

    if alpha < 1e-10: return l1_loss
    if alpha > 1 - 1e-10: return msssim_loss

    total_loss = alpha * msssim_loss + (1 - alpha) * l1_loss
    return total_loss