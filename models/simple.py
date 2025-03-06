import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) + out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, z_size, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(z_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, z_size)
    
    def forward(self, z, t):
        za = z
        za = F.relu(self.lin1(za, t))
        za = F.relu(self.lin2(za, t))
        za = F.relu(self.lin3(za, t))
        za = F.relu(self.lin4(za, t))
        return self.lin_mean(za)
    

class ConditionalModelWAUX(nn.Module):
    def __init__(self, in_size, n_steps, out_size):
        super().__init__()
        self.lin1 = ConditionalLinear(in_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, out_size)
    
    def forward(self, z, a, t):
        za = torch.cat((z, a), 1)
        za = F.relu(self.lin1(za, t))
        za2 = F.relu(self.lin2(za, t))
        za2 = F.relu(self.lin3(za2, t)) + za
        za2 = F.relu(self.lin4(za2, t))
        return self.lin_mean(za2)


class DoubleConditionalModel(nn.Module):
    def __init__(self, z_size, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(z_size, 128, n_steps) # TODO: FIX - should be number of classes
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin5 = ConditionalLinear(128, 128, n_steps)
        self.lin6 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, z_size)
    
    def forward(self, z, a, t):
        za = z
        za = F.relu(self.lin1(za, a))
        za1 = za
        za = F.relu(self.lin2(za, t))
        za = F.relu(self.lin3(za, a) + za1)
        za3 = za
        za = F.relu(self.lin4(za, t))
        za = F.relu(self.lin5(za, a) + za3)
        za = F.relu(self.lin6(za, t))
        return self.lin_mean(za)


class TimeClassifier(nn.Module):
    def __init__(self, in_size, out_size, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(in_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, out_size)
    
    def forward(self, z, t):
        za = z
        za = F.relu(self.lin1(za, t))
        za = F.relu(self.lin2(za, t))
        za = F.relu(self.lin3(za, t))
        za = F.relu(self.lin4(za, t))
        return self.lin_mean(za)


class MultiWayLinear(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_sizes, sigmoid=False, bn=False, dropout=None):
        super().__init__()
        self.sigmoid = sigmoid
        self.use_bn = bn
        self.dropout = dropout

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        if isinstance(out_sizes, int):
            out_sizes = [out_sizes]
        self.lins = nn.ModuleList([nn.Linear(in_size, hidden_sizes[0])] + [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        self.lin_outs = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_size) for out_size in out_sizes])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in hidden_sizes])
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.leaky_relu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        res = [lin_out(x) for lin_out in self.lin_outs]
        if self.sigmoid:
            res = [F.sigmoid(out) for out in res]
        return res if len(self.lin_outs) > 1 else res[0]