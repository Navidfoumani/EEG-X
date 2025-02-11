import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)]
    
class PositionalEmbedding1D(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding1D, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def positional_encoding_2d(coords, d):
    """
    Generate sinusoidal positional embeddings for a list of (x, y) coordinates.
    
    Args:
        coords: List of (x, y) tuples representing positions.
        d: Embedding dimension (must be even).

    Returns:
        pos_embedding: (N, d) matrix of positional embeddings.
    """
    N = len(coords)
    pos_embedding = np.zeros((N, d))
    
    for idx, (x, y) in enumerate(coords):
        for k in range(d // 4):  # Half for x, half for y
            pos_embedding[idx, 4*k] = np.sin(x / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 1] = np.cos(x / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 2] = np.sin(y / (5 ** (4*k / d)))
            pos_embedding[idx, 4*k + 3] = np.cos(y / (5 ** (4*k / d)))
    
    return pos_embedding


class elec_location_Embedding(nn.Module):
    def __init__(self, d_model):
        super(elec_location_Embedding, self).__init__()
        coords = pd.read_csv("notebooks/electrode_positions.csv")
        coords['x'] = (coords['x']+0.8) *5
        coords['y'] = (coords['y']+1)* 5
        new_coords = [(x, y) for x, y in zip(coords['x'], coords['y'])]
        pos_embedding = positional_encoding_2d(new_coords, d_model)

        epoch14_electrode_embeddings = match_epoch14_electrodes(coords,pos_embedding)
        self.epoch14_electrode_embeddings = torch.tensor(epoch14_electrode_embeddings).float().cuda()

        THU19_electrode_embeddings = match_TUH19_electrodes(coords,pos_embedding)
        self.THU19_electrode_embeddings = torch.tensor(THU19_electrode_embeddings).float().cuda()

        BCI22_electrodes__embeddings = match_BCI22_electrodes(coords,pos_embedding)
        self.BCI22_electrodes__embeddings = torch.tensor(BCI22_electrodes__embeddings).float().cuda()

    def forward(self, x):
        if x.shape[1] == 14: # 14 electrodes
            embedding = self.epoch14_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 19: # 19 electrodes
            embedding = self.THU19_electrode_embeddings.unsqueeze(1).repeat(1, x.size(2), 1)
        elif x.shape[1] == 22: # 22 electrodes 
            embedding = self.BCI22_electrodes__embeddings.unsqueeze(1).repeat(1, x.size(2), 1)    
        return embedding


def match_epoch14_electrodes(coords, pos_embedding):
    epoch14_electrodes = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']
    coords['label'] = coords['label'].str.lower()
    epoch14_electrode_indices = coords[coords['label'].isin(epoch14_electrodes)].index.tolist()
    epoch14_electrode_embeddings = pos_embedding[epoch14_electrode_indices]
    return epoch14_electrode_embeddings


def match_BCI22_electrodes(coords, pos_embedding):
    BCI22_electrodes = ['fz', 
                          'fc3', 'fc1', 'fcz', 'fc2', 'fc4',
                           'c5', 'c3', 'c1', 'cz', 'c2', 'c4', 'c6',
                             'cp3', 'cp1', 'cpz', 'cp2', 'cp4',
                               'p1', 'pz', 'p2',
                                 'poz']
    coords['label'] = coords['label'].str.lower()
    BCI22_electrodes_indices = coords[coords['label'].isin(BCI22_electrodes)].index.tolist()
    BCI22_electrodes_embeddings = pos_embedding[BCI22_electrodes_indices]
    return BCI22_electrodes_embeddings


def match_TUH19_electrodes(coords, pos_embedding):
    TUH_19_electrodes = ['fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1',
                         'o2', 'f7', 'f8', 't7', 't8', 'p6', 'p5',
                         'fz', 'cz', 'pz']
    coords['label'] = coords['label'].str.lower()
    TUH_19_electrodes_indices = coords[coords['label'].isin(TUH_19_electrodes)].index.tolist()
    TUH_19_electrodes_indices = pos_embedding[TUH_19_electrodes_indices]
    return TUH_19_electrodes_indices