import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#UNET
norm_dict = {
    'batch': nn.BatchNorm2d,
    'instance': nn.InstanceNorm2d,
    'layer': nn.LayerNorm
}

activations_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
}
#Mini Convolutional Unet for MNIST
class UNetConvBlock(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_w=3, activation='relu', norm='instance', depth=2):
        super().__init__()
        assert (kernel_w - 1) % 2 == 0
        assert depth >= 2

        padding = int(math.floor(kernel_w / 2))

        self.depth = depth
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.activation = activations_dict[activation]()

        for d in range(depth):
            if d == 0:
                self.convs.append(nn.Conv2d(in_chs, out_chs, kernel_w, padding=padding))
                self.norms.append( norm_dict[norm](in_chs))
            else:
                self.convs.append(nn.Conv2d(out_chs, out_chs, kernel_w, padding=padding))
                self.norms.append( norm_dict[norm](out_chs))

        if in_chs == out_chs:
            self.skip_transform = nn.Identity()
        else:
            self.skip_transform = nn.Conv2d(in_chs, out_chs, 1)

    def forward(self, x):
        z = x
        for d, (norm, conv) in enumerate(zip(self.norms, self.convs)):
            z = conv(norm(z))
            if d < self.depth - 1:
                z = self.activation(z)

        z = self.skip_transform(x) + z
        z = self.activation(z)
        return z


class UNetEncoder(nn.Module):
    def __init__(self, chs=(3,64, 128, 256), activation='relu', conv_block_depth=2):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.conv_blocks.append(UNetConvBlock(chs[i], chs[i+1], activation=activation, depth=conv_block_depth))

        self.down_pool = nn.MaxPool2d(2)

    def forward(self, x):
        feature_maps = []
        for i, block in enumerate(self.conv_blocks):
            x = block(x)

            if i < (len(self.conv_blocks) - 1):
                feature_maps.append(x)
                x = self.down_pool(x)
            else:
                return x, feature_maps

class UNetDecoder(nn.Module):
    def __init__(self, chs=(64, 128, 256), activation='relu', conv_block_depth=2):
        super().__init__()
        self.chs = chs
        self.upsampler = nn.ModuleList()
        ch_reversed = tuple(reversed(chs))
        for i in range(len(chs) - 1):
            self.upsampler.append(nn.ConvTranspose2d(ch_reversed[i], ch_reversed[i+1], 2, 2))

        self.conv_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.conv_blocks.append(UNetConvBlock(ch_reversed[i], ch_reversed[i+1],
                                                  activation=activation, depth=conv_block_depth))

    def forward(self, x, encoder_features):
        for i, (upsampler, conv_block) in enumerate(zip(self.upsampler, self.conv_blocks)):
            x = upsampler(x)
            encoder_features_i = encoder_features[len(encoder_features) - 1 - i]
            x = torch.cat([x, encoder_features_i], dim=1)
            x = conv_block(x)
        return x


class MiniUNet(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256), activation='elu', conv_block_depth=2, sigmas=None):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.encoder =  UNetEncoder((in_channels, ) + channels, conv_block_depth=conv_block_depth)
        self.decoder =  UNetDecoder(channels, conv_block_depth=conv_block_depth)
        self.head = nn.Conv2d(channels[0], in_channels, 1, padding=0)
        self.activation = activations_dict[activation]()

    def forward(self, x, t):

        encoder_out, encoder_maps = self.encoder(x)
        decoder_out = self.decoder(encoder_out, encoder_maps)
        x = self.activation(decoder_out)
        x = self.head(x)

        #Technique3 in Improved Techniques for Training ...
        if self.sigmas is not None:
            sigmas_t = self.sigmas[t].view(x.shape[0], *([1] * len(x.shape[1:])))
            x = x / sigmas_t

        return x

def get_MNIST_UNet(sigmas=None):
    return MiniUNet(1, sigmas=sigmas)
def get_CIFAR_UNet(sigmas=None):
    return MiniUNet(3, channels=(64, 128, 256, 512), sigmas=sigmas)

def get_CIFAR_UNet_Large(sigmas=None):
    return MiniUNet(3, channels=(64, 128, 256, 512), conv_block_depth=3, sigmas=sigmas)
