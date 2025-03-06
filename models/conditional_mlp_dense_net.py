import torch
import torch.nn as nn

from .modules.helpers import Residual
from .modules.attention import PreNorm
from .modules.embeddings import SineCosinePositionEmbeddings


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, input_size, in_channels, growth_rate, time_emb_dim=None):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck1 = nn.Sequential(
            nn.BatchNorm1d(in_channels * input_size),
            nn.ReLU(),
            nn.Linear(in_channels * input_size, inner_channel * input_size, bias=False),
        )
        self.bottle_neck2 = nn.Sequential(
            nn.BatchNorm1d(inner_channel * input_size),
            nn.ReLU(),
            nn.Linear(inner_channel * input_size, growth_rate * input_size, bias=False)
        )

        if time_emb_dim is not None:
            self.time_embedding_generator = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, inner_channel * input_size))

    def forward(self, x, t=None):
        new_x = self.bottle_neck1(x)
        if t is not None:
            new_x = new_x + self.time_embedding_generator(t)
        new_x = self.bottle_neck2(new_x)
        return torch.cat([x, new_x], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, input_size, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm1d(in_channels * input_size),
            nn.Linear(in_channels * input_size, out_channels * input_size, bias=False),
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class ConditionalMLPDenseNet(nn.Module):
    def __init__(self, input_size, block, nblocks, growth_rate=12, reduction=0.5, a_size=30):
        super().__init__()
        self.growth_rate = growth_rate
        self.input_size = input_size

        initial_time_dim = 4
        self.time_dim = 8 # why 4? should comment on this
        self.time_mlp = nn.Sequential(
            SineCosinePositionEmbeddings(initial_time_dim),
            nn.Linear(initial_time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.lin1 = nn.Linear(self.input_size, inner_channels * input_size)

        self.downs = nn.ModuleList([])

        for index in range(len(nblocks) - 1):
            layers = []
            # self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            layers.append(self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            # cross attention 
            layers.append(Residual(PreNorm(inner_channels * self.input_size, nn.Linear(inner_channels * self.input_size + a_size, inner_channels * self.input_size))))

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            # self.features.add_module("transition_layer_{}".format(index), Transition(self.input_size, inner_channels, out_channels))
            layers.append(Transition(self.input_size, inner_channels, out_channels))
            self.downs.append(nn.ModuleList(layers))
            inner_channels = out_channels

        layers = []
        # self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        layers.append(self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]

        layers.append(Residual(PreNorm(inner_channels * self.input_size, nn.Linear(inner_channels * self.input_size + a_size, inner_channels * self.input_size))))

        # self.features.add_module('bn', nn.BatchNorm1d(inner_channels * self.input_size))
        # self.features.add_module('relu', nn.ReLU())
        layers.append(nn.Sequential(nn.BatchNorm1d(inner_channels * self.input_size), 
            nn.ReLU(),
            nn.Linear(inner_channels * self.input_size, self.input_size)))
        self.downs.append(nn.ModuleList(layers))

    def forward(self, x, a, t):
        t = self.time_mlp(t)
        output = self.lin1(x)
        # output = self.features(output)
        for dense_layers, xattn, transition in self.downs:
            output = dense_layers(output, t)
            output = xattn(output, a)
            output = transition(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = mySequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(self.input_size, in_channels, self.growth_rate, self.time_dim))
            in_channels += self.growth_rate
        return dense_block

def conditional_mlp_densenet_mini(input_size, a_size):
    return ConditionalMLPDenseNet(input_size, Bottleneck, [2, 2, 2, 2], growth_rate=12, a_size=a_size)
