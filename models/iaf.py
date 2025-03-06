# taken from https://github.com/pclucas14/iaf-vae/blob/master/layers.py
import torch
import torch.nn.functional as F
from torch import nn

from .made import MADE


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=False) # TODO: Should I use natural ordering?
        
    def forward(self, x, h):
        return self.net(x, h)


class IAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim_z, dim_h, parity, net_class=ARMLP, nh=128):
        super().__init__()
        self.dim = dim_z
        self.net = net_class(dim_z, dim_z*2, nh)
        self.parity = parity

    def forward(self, z, h):
        # here we see that we are evaluating all of x in parallel, so density estimation will be fast
        st = self.net(z, h)
        s, t = st.split(self.dim, dim=1) 
        var = torch.sigmoid(s)
        log_var = torch.log(var)

        x = var * z + (1 - var) * t
        # reverse order, so if we stack MAFs correct things happen
        x = x.flip(dims=(1,)) if self.parity else x
        # log_det = torch.sum(s, dim=1)
        log_det = torch.sum(log_var, dim=1)
        return x, log_det
    

class ConditionalMADE(nn.Module):
    def __init__(self, in_size, n_steps, out_size):
        super().__init__()
        self.net = ARMLP(in_size, out_size, 128)
        self.embed = nn.Embedding(n_steps, 8) # 8 + 2 = 10
        self.embed.weight.data.uniform_()
    
    def forward(self, z, a, t):
        gamma = self.embed(t)
        su = self.net(z, torch.cat((a, gamma.view(-1, 8)), 1))
        return su


class StackedMADE(nn.Module):
    def __init__(self, in_size, n_steps, out_size):
        super().__init__()
        self.net = nn.ModuleList([ARMLP(in_size, out_size, 24) for _ in range(n_steps)])
    
    def forward(self, z, a, t):
        if t[0] != t[1]:
            raise NotImplementedError
        return self.net[t[0].item()](z, a)