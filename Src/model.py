import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel=3, down_mode='max', pad_mode='zero'):
        super().__init__()
        
        if down_mode=='max':
            self.down = nn.MaxPool2d(2)
        elif down_mode=='avg':
            self.down = nn.AvgPool2d(2)
        elif down_mode=='stride':
            self.down = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=2)
        else:
            assert False, 'unknown downsampling mode'
        
        self.convblock = nn.Sequential(
            self.down,
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=kernel, padding=kernel//2, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.convblock(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_chan, out_chan, skip_chan, kernel=3, up_mode='bilinear', pad_mode='zero'):
        super().__init__()
   
        if up_mode=='stride':
            self.up = nn.ConvTranspose2d(out_chan, out_chan, kernel_size=2, stride=2)
        elif up_mode in ['nearest', 'bilinear']:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode)
        else:
            assert False, 'unknown upsampling mode'
            
        self.convblock = nn.Sequential(
            nn.BatchNorm2d(in_chan + skip_chan),
            nn.Conv2d(in_chan + skip_chan, out_chan, kernel_size=kernel, padding=kernel//2, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, padding=0, padding_mode=pad_mode),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.convblock(x)


class model_unet(nn.Module):
    def __init__(self, 
                 input_shape=32,
                 output_shape=1,
                 down_filters=[16,32,64,128],
                 up_filters=[16,32,64,128],
                 skip_filters=[16,16,16,16],
                 down_kernels=[3,3,3,3],
                 up_kernels=[3,3,3,3],
                 skip_kernels=[1,1,1,1],
                 up_mode='bilinear',
                 down_mode='max',
                 pad_mode='reflection',
                ):
        super().__init__()

        assert len(down_filters) == len(up_filters) == len(skip_filters) == len(down_kernels) == len(up_kernels) == len(skip_kernels)

        self.depth = len(down_filters)
        
        self.down_layers, self.up_layers, self.skip_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(),
        self.down_filters, self.up_filters, self.skip_filters = down_filters, up_filters, skip_filters
        
        for idx in range(self.depth):
            
            self.down_layers.append(
                DownBlock(input_shape, down_filters[idx], kernel=down_kernels[idx], down_mode=down_mode, pad_mode=pad_mode) if idx==0
                else DownBlock(down_filters[idx-1], down_filters[idx], kernel=down_kernels[idx], down_mode=down_mode, pad_mode=pad_mode)
            )
            self.up_layers.append(
                UpBlock(down_filters[-1], up_filters[idx], skip_filters[idx], kernel=up_kernels[idx], up_mode=up_mode, pad_mode=pad_mode) if idx==self.depth-1
                else UpBlock(up_filters[idx+1], up_filters[idx], skip_filters[idx], kernel=up_kernels[idx], up_mode=up_mode, pad_mode=pad_mode)
            )
            if skip_filters[idx]!=0:
                self.skip_layers.append(
                    nn.Sequential(
                        nn.Conv2d(input_shape, skip_filters[idx], kernel_size=skip_kernels[idx], padding=skip_kernels[idx]//2, padding_mode=pad_mode) if idx==0 
                        else nn.Conv2d(down_filters[idx-1], skip_filters[idx], kernel_size=skip_kernels[idx], padding=skip_kernels[idx]//2, padding_mode=pad_mode),
                        nn.BatchNorm2d(skip_filters[idx]),
                        nn.LeakyReLU()))
            else:
                self.skip_layers.append(None)

        self.out_conv = nn.Sequential(
            nn.Conv2d(up_filters[0], 4, kernel_size=3, padding=3//2, padding_mode=pad_mode),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(4, output_shape, kernel_size=3, padding=3//2, padding_mode=pad_mode),
            nn.LeakyReLU()
        )

    def forward(self, x):
        temp_skip = []
        
        # Down 
        for idx, block in enumerate(self.down_layers):
            temp_skip.append(
                None if self.skip_filters[idx]==0
                else self.skip_layers[idx](x)
            )
            x = block(x)

        # Up & skip connection
        for idx, block in enumerate(reversed(self.up_layers)):
            x = block(x, temp_skip[self.depth-idx-1])

        x = self.out_conv(x)
        
        return x