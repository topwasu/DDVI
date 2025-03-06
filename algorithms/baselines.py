import numpy as np
import umap
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils import visualize_cluster_latents

class BaseBaseline:
    def __init__(self, config):
        self.seed = config.seed
        self.config = config

    def get_numpy_data(self, dataloader):
        all_genotypes = []
        for data in dataloader:
            all_genotypes.append(data.cpu().numpy())
        
        all_genotypes = np.concatenate(all_genotypes, 0)
        return all_genotypes
    
    def eval_label(self, dataloader):
        pcs = self.eval(dataloader)
        predicted = KMeans(n_clusters=20, random_state=self.seed, n_init=1).fit_predict(pcs)
        visualize_cluster_latents(pcs, os.path.join(self.config.save_folder, 'latents_clustered.png'), predicted)
        return predicted


class PCAModel(BaseBaseline):    
    def eval(self, dataloader):
        all_genotypes = self.get_numpy_data(dataloader)
        pcs = PCA(n_components=2, svd_solver='randomized', random_state=self.seed).fit_transform(all_genotypes)
        return pcs
    

class TSNEModel(BaseBaseline):    
    def eval(self, dataloader):
        all_genotypes = self.get_numpy_data(dataloader)
        pcs = TSNE(n_components=2, perplexity=30, random_state=self.seed).fit_transform(all_genotypes)
        return pcs
    

class UMAPModel(BaseBaseline):    
    def eval(self, dataloader):
        all_genotypes = self.get_numpy_data(dataloader)
        pcs = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, verbose=True, random_state=self.seed).fit_transform(all_genotypes)
        return pcs
