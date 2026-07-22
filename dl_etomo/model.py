"""
model.py
========
2D U-Net architecture for Deep Image Prior (DIP) tomographic reconstruction.

Based on: O. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.

The architecture is fully configurable: number of levels, filter counts, kernel
sizes, skip-connection widths, and up/downsampling modes can all be set at
construction time.

Public API
----------
model_unet   : configurable 2D U-Net nn.Module
DownBlock    : single encoder stage (pool + two conv layers)
UpBlock      : single decoder stage (upsample + skip-cat + two conv layers)
"""

import torch
from torch import nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """
    Encoder block: downsample → Conv2d → BN → LeakyReLU × 2.

    Parameters
    ----------
    in_chan : int
        Number of input feature channels.
    out_chan : int
        Number of output feature channels.
    kernel : int
        Kernel size for the second convolution (first is always 3×3).
    down_mode : {'max', 'avg', 'stride'}
        Downsampling strategy.
        - 'max'    : 2×2 max-pooling
        - 'avg'    : 2×2 average-pooling
        - 'stride' : strided 3×3 convolution (learnable)
    pad_mode : str
        Padding mode passed to ``nn.Conv2d`` (e.g. 'zeros', 'reflect').
    """

    def __init__(self, in_chan, out_chan, kernel=3, down_mode='max', pad_mode='zeros'):
        super().__init__()

        if down_mode == 'max':
            down = nn.MaxPool2d(2)
        elif down_mode == 'avg':
            down = nn.AvgPool2d(2)
        elif down_mode == 'stride':
            down = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=2)
        else:
            raise ValueError(f"Unknown down_mode: '{down_mode}'. Use 'max', 'avg', or 'stride'.")

        self.convblock = nn.Sequential(
            down,
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=kernel,
                      padding=kernel // 2, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """
    Decoder block: upsample → concat skip → Conv2d → BN → LeakyReLU × 2.

    Parameters
    ----------
    in_chan : int
        Channels coming from the deeper decoder level.
    out_chan : int
        Output channels after this block.
    skip_chan : int
        Channels from the corresponding encoder skip connection.
    kernel : int
        Kernel size for the convolutions.
    up_mode : {'bilinear', 'nearest', 'stride'}
        Upsampling strategy.
        - 'bilinear' / 'nearest' : interpolation
        - 'stride'               : transposed convolution
    pad_mode : str
        Padding mode passed to ``nn.Conv2d``.
    """

    def __init__(self, in_chan, out_chan, skip_chan, kernel=3,
                 up_mode='bilinear', pad_mode='zeros'):
        super().__init__()

        self.up_mode = up_mode
        if up_mode == 'stride':
            self.up = nn.ConvTranspose2d(out_chan, out_chan, kernel_size=2, stride=2)
        elif up_mode in ('nearest', 'bilinear'):
            self._up_kwargs = (dict(mode=up_mode) if up_mode == 'nearest'
                               else dict(mode=up_mode, align_corners=False))
        else:
            raise ValueError(f"Unknown up_mode: '{up_mode}'. Use 'bilinear', 'nearest', or 'stride'.")

        self.convblock = nn.Sequential(
            nn.BatchNorm2d(in_chan + skip_chan),
            nn.Conv2d(in_chan + skip_chan, out_chan, kernel_size=kernel,
                      padding=kernel // 2, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
        )

    def forward(self, x, skip):
        if self.up_mode == 'stride':
            x = self.up(x)
        elif skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], **self._up_kwargs)
        else:
            x = F.interpolate(x, scale_factor=2, **self._up_kwargs)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.convblock(x)


class model_unet(nn.Module):
    """
    Configurable 2D U-Net for DIP tomographic reconstruction.

    Parameters
    ----------
    input_shape : int
        Number of input noise channels.
    output_shape : int
        Number of output channels (1 for grayscale reconstruction).
    down_filters : tuple of int
        Feature-map counts at each encoder level.
    up_filters : tuple of int
        Feature-map counts at each decoder level (same length as down_filters).
    skip_filters : tuple of int
        Channels in each skip connection (0 disables skip at that level).
    down_kernels, up_kernels, skip_kernels : tuple of int
        Kernel sizes at each level for the encoder, decoder, and skip 1×1 convs.
    up_mode : str
        Upsampling strategy passed to ``UpBlock``.
    down_mode : str
        Downsampling strategy passed to ``DownBlock``.
    pad_mode : str
        Padding mode for all convolutions.
    out_kernel : int
        Kernel size of the final output convolution.
    """

    def __init__(self,
                 input_shape=32,
                 output_shape=1,
                 down_filters=(16, 32, 64, 128),
                 up_filters=(16, 32, 64, 128),
                 skip_filters=(16, 16, 16, 16),
                 down_kernels=(3, 3, 3, 3),
                 up_kernels=(3, 3, 3, 3),
                 skip_kernels=(1, 1, 1, 1),
                 up_mode='bilinear',
                 down_mode='max',
                 pad_mode='reflect',
                 out_kernel=1):
        super().__init__()

        assert len(down_filters) == len(up_filters) == len(skip_filters) \
               == len(down_kernels) == len(up_kernels) == len(skip_kernels)

        self.depth = len(down_filters)
        self.down_filters = list(down_filters)
        self.up_filters = list(up_filters)
        self.skip_filters = list(skip_filters)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for idx in range(self.depth):
            in_d = input_shape if idx == 0 else down_filters[idx - 1]
            self.down_layers.append(
                DownBlock(in_d, down_filters[idx], kernel=down_kernels[idx],
                          down_mode=down_mode, pad_mode=pad_mode)
            )
            in_u = (down_filters[-1] if idx == self.depth - 1
                    else up_filters[idx + 1])
            self.up_layers.append(
                UpBlock(in_u, up_filters[idx], skip_filters[idx],
                        kernel=up_kernels[idx], up_mode=up_mode, pad_mode=pad_mode)
            )
            if skip_filters[idx] != 0:
                in_s = input_shape if idx == 0 else down_filters[idx - 1]
                self.skip_layers.append(nn.Sequential(
                    nn.Conv2d(in_s, skip_filters[idx],
                              kernel_size=skip_kernels[idx],
                              padding=skip_kernels[idx] // 2,
                              padding_mode=pad_mode),
                    nn.BatchNorm2d(skip_filters[idx]),
                    nn.LeakyReLU(),
                ))
            else:
                self.skip_layers.append(None)

        self.out_conv = nn.Sequential(
            nn.Conv2d(up_filters[0], 4, kernel_size=out_kernel,
                      padding=out_kernel // 2, padding_mode=pad_mode),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(4, output_shape, kernel_size=out_kernel,
                      padding=out_kernel // 2, padding_mode=pad_mode),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (1, input_shape, H, W)

        Returns
        -------
        torch.Tensor, shape (1, output_shape, H, W)
        """
        temp_skip = []

        # Encoder: collect skip-connection feature maps
        for idx, block in enumerate(self.down_layers):
            temp_skip.append(
                None if self.skip_filters[idx] == 0
                else self.skip_layers[idx](x)
            )
            x = block(x)

        # Decoder: upsample and fuse with skip connections (reverse order)
        for idx, block in enumerate(reversed(self.up_layers)):
            x = block(x, temp_skip[self.depth - idx - 1])

        return self.out_conv(x)
