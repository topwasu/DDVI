import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from matplotlib import colors
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from cyvcf2 import VCF
from bokeh.io import output_notebook, export_png
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Category20b, Purples, Greens, YlOrBr, YlOrRd, PuOr, RdGy
from bokeh.plotting import figure, show, output_file


log = logging.getLogger(__name__)


import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def visualize_mnist_latent(res, save_path):
    plt.clf()
    res = res.reshape((-1, 2))
    test_dataset = MNIST('.', train=False, download=True)
    targets = test_dataset.targets.numpy()
    if len(res) < 10000:
        indices = np.concatenate([np.nonzero(targets == 0), np.nonzero(targets == 1)], axis=None)
        targets = targets[indices]
    plt.scatter(res[:,0], res[:,1], s = 5, alpha = 0.8, c = targets, cmap = colors.ListedColormap(['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:olive', 'black', 'tab:pink', 'tab:gray', 'tab:brown']))
    plt.savefig(save_path, bbox_inches='tight')


def visualize_latent(res, save_path, targets=None):
    plt.clf()
    res = res.reshape((-1, 2))
    if targets is None:
        plt.scatter(res[:,0], res[:,1], s = 5, alpha = 0.8)
    else:
        plt.scatter(res[:,0], res[:,1], s = 5, alpha = 0.8, c = targets, cmap = colors.ListedColormap(['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:olive', 'black', 'tab:pink', 'tab:gray', 'tab:brown']))
    plt.savefig(save_path, bbox_inches='tight')


def visualize_cluster_latents(res, save_path, targets=None):
    plt.clf()
    res = res.reshape((-1, 2))
    if targets is None:
        plt.scatter(res[:,0], res[:,1], s = 5, alpha = 0.8)
    else:
        plt.scatter(res[:,0], res[:,1], s = 5, alpha = 0.8, c = targets)
    plt.savefig(save_path, bbox_inches='tight')


def save_grid(img, save_path):
    plt.clf()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig(save_path, bbox_inches='tight')


def visualize_mnist_rec(model, res, save_path):
    rec = model.get_rec(torch.tensor(res[:64]))
    grid = make_grid(rec.view(64, 1, 28, 28), ncol=8)
    save_grid(grid, save_path)


def visualize_mnist(res, save_path):
    # grid = make_grid(res.view(-1, 1, 28, 28), ncol=8)
    grid = make_grid(res.view(-1, 1, 28, 28), nrow=10)
    save_grid(grid, save_path)


def gaussian_kernel(a, b, h=0.3):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = ((a_core - b_core)).pow(2).mean(2)/depth/2
    print(numerator.size())
    return torch.exp(-numerator)


def mmd_func(a, b, h=0.3):
    return torch.sqrt(gaussian_kernel(a, a, h).mean() + gaussian_kernel(b, b, h).mean() - 2*gaussian_kernel(a, b, h).mean())

def mmd_loss(x, gen_x, sigma = [2, 5, 10, 20, 40, 80]):
    # concatenation of the generated images and images from the dataset
    # first 'N' rows are the generated ones, next 'M' are from the data
    X = torch.cat([gen_x, x], axis=0)
    # dot product between all combinations of rows in 'X'
    XX = torch.matmul(X, X.T)
    # dot product of rows with themselves
    X2 = torch.sum(X * X, 1, keepdims=True)
    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * X2.T
    # scaling constants for each of the rows in 'X'
    s = makeScaleMatrix(gen_x.shape[0], x.shape[0])
    # scaling factors of each of the kernel values, corresponding to the
    # exponent values
    S = torch.matmul(s, s.T)
    print(S.size())
    loss = 0
    # for each bandwidth parameter, compute the MMD value and add them all
    for i in range(len(sigma)):
        # kernel values for each combination of the rows in 'X' 
        kernel_val = torch.exp(1.0 / sigma[i] * exponent)
        loss += torch.sum(S * kernel_val)
    return torch.sqrt(loss)


def makeScaleMatrix(num_gen, num_orig):
    # first 'N' entries have '1/N', next 'M' entries have '-1/M'
    s1 =  torch.full((num_gen, 1), 1.0 / num_gen)
    s2 = -torch.full((num_orig, 1), 1.0 / num_orig)
    return torch.cat([s1, s2], axis=0)


def slow_mmd_loss(x, gen_x, sigma = [2, 5, 10, 20, 40, 80], is_sigma_squared = True):
    if not is_sigma_squared:
        sigma_squared = np.asarray(sigma) ** 2
    else:
        sigma_squared = sigma

    squared_loss = 0
    n = x.size()[0]
    m = gen_x.size()[0]
    for i in range(n):
        diff = torch.sum((x-x[i]) ** 2, -1)
        for s in sigma_squared:            
            squared_loss = squared_loss + torch.sum(torch.exp(-diff / (2 * s))) / n / n 
    print('Hi')
    for i in range(n):
        diff = torch.sum((gen_x-x[i]) ** 2, -1)
        for s in sigma_squared:            
            squared_loss = squared_loss - 2 * torch.sum(torch.exp(-diff / (2 * s))) / n / m 
    print('Hi2')
    for i in range(m):
        diff = torch.sum((gen_x-gen_x[i]) ** 2, -1)
        for s in sigma_squared:            
            squared_loss = squared_loss + torch.sum(torch.exp(-diff / (2 * s))) / m / m
    
    return torch.sqrt(squared_loss)


def visualize_1kgenome(res, save_path):
    # Reference: https://github.com/diazale/1KGP_dimred/blob/master/Genotype_dimred_demo.ipynb

    folder = '/home/wp237/genomics_data'
    vcf_name = f'{folder}/1000-genomes-snp/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz'
    pop_desc_name = f'{folder}/1000-genomes-snp/20131219.populations.tsv'
    pop_file_name = f'{folder}/1000-genomes-snp/affy_samples.20141118.panel'

    # get samples
    individuals = VCF(vcf_name).samples

    # get pop by continent
    name_by_code = {}
    pop_by_continent = defaultdict(list)
    with open(pop_desc_name ,'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split('\t')
        if split_line[0] in ['Population Description','Total','']:  # header or footer
            continue
        name_by_code[split_line[1]] = split_line[0]
        pop_by_continent[split_line[2]].append(split_line[1])
    continents = list(pop_by_continent.keys())
    pops=[]
    for continent in continents:
        pops.extend(pop_by_continent[continent])

    # get pop by individ and individs by pop
    population_by_individual = defaultdict(int)
    individuals_by_population = defaultdict(list)
    with open(pop_file_name ,'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        if split_line[0] == 'sample':  # header line
            continue
        sample_name = split_line[0]
        population_name = split_line[1]
        population_by_individual[sample_name] = population_name
        individuals_by_population[population_name].append(sample_name) 

    # get indices of pop members by pop
    indices_of_population_members = defaultdict(list)
    for idx, individual in enumerate(individuals):
        try:
            indices_of_population_members[population_by_individual[individual]].append(idx)
        except KeyError: # We do not have population info for this individual
            continue
            
    targets = []
    for ind in individuals:
        pop = population_by_individual[ind]
        targets.append(pop + ' - ' + name_by_code[pop])
        
    # Assign colours to each population, roughly themed according to continent
    # The Category20b palette has a bunch of groups of 4 shades in the same colour range
    color_dict = {}
    for i, cont in enumerate(continents): 
        for j, pop in enumerate(pop_by_continent[cont]):
            color_dict[pop] = Category20b[20][4*i+j%4]

    # Colour palette above only really supports groups of 4 so we have to manually specify a few colours for the 5th/6th
    # members of a group

    color_dict['CHS'] = Purples[9][4]# purple
    color_dict['STU'] = Greens[9][6] # green
    color_dict['LWK'] = PuOr[11][-1] # brown
    color_dict['MSL'] = PuOr[11][-2] # rusty brown
    color_dict['YRI'] = PuOr[11][-3] # cappucino w/ extra milk (stirred)
    color_dict['CEU'] = RdGy[11][-3]

    # --------- Finish prepping ----------
    name = '1kgenome'
    
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("label", "@label"),
    ]
    p = figure(plot_width=1350, plot_height=800, tooltips=TOOLTIPS)
    p.title.text = name
    for pop in pops:
        pop_indices = indices_of_population_members[pop]
        source = ColumnDataSource(
            data={'x': res[pop_indices, 0], 'y': res[pop_indices, 1], 'label': [name_by_code[pop]] * len(pop_indices)}
        )
        p.circle('x', 'y', legend_label=name_by_code[pop], color = color_dict[pop], source=source)
    p.legend.label_text_font_size = '8pt'
    p.legend.click_policy="hide"
    p.add_layout(p.legend[0], 'right')
    
    export_png(p, filename=save_path)


def visualize_eurasia(res, save_path):
    folder = '/home/wp237/popgen_course_data'
    with open(f'{folder}/AllEurasia.poplist.txt') as f:
        lines = f.readlines()
    all_eurasia = [line.strip() for line in lines]

    ind_df = pd.read_csv(f'{folder}/HumanOrigins_FennoScandian_small.ind', names=['sample_id', 'gender', 'status'], sep="\s+") 
    ind_df = ind_df.astype({'sample_id': 'str', 'gender': 'str', 'status': 'str'})
    status_arr = ind_df['status'][ind_df['status'].isin(all_eurasia)].to_numpy()

    # --------- Finish prepping ----------
    name = 'eurasia'
    
    
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    markers = ['circle', 'square', 'triangle', 'plus', 'star', 'hex', 'hex_dot', 'cross', 'x', 'y', 'diamond_dot', 'diamond', 'circle_dot', 'square_dot']
    
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("label", "@label"),
    ]
    p = figure(plot_width=1500, plot_height=800, tooltips=TOOLTIPS)
    p.title.text = name
    
    items = []
    legenddict = {}
    for idx, pop in enumerate(all_eurasia):
        ind = np.where(status_arr == pop)[0]
        source = ColumnDataSource(
            data={'x': res[ind, 0], 'y': res[ind, 1], 'label': [pop] * len(ind)}
        )
        legenddict[pop] = p.scatter('x', 'y', legend_label=pop, color=colors[idx // len(markers)], marker=markers[idx % len(markers)], source=source)
        items.append((pop, [legenddict[pop]]))
    p.legend.visible=False
    for i in range(0, len(items), 30):
        legend1 = Legend(
            items=items[i:i+30],
            location=(0, 10 + i / 6))

        p.add_layout(legend1, 'right')
    p.legend.click_policy="hide"
    
    export_png(p, filename=save_path)