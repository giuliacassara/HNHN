
'''
using hypergraph representations for document classification.
'''
import _init_paths
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

import argparse
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Sequential, GCNConv  # noqa
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepSet(MessagePassing):
    '''
    The class returns a hyperedge-level representation of an input hypergraph. 
    Using the sum-pooling layer results in the invariance of the architecture to 
    node permutation inside each hyperedge. This vector representation is the used for
    prediction, adding a GCN or GAT layer to predict each node's label.

    in_channels (int): Size of each input sample, or :obj:`-1` to derive 
    the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.

        Shapes:
        - **input:**
          node features {V}|, F)`,
          hyperedge indices {V}|, {E}|)`,
          hyperedge weights |E|
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum') #  "Sum" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
        self.sigmoid = torch.nn.Sigmoid()

    
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i):
        # x_i has shape [E, in_channels]
        output = self.mlp(x_i)
        return self.sigmoid(output)
        

class GCNConv(torch.nn.Module):
    def __init__(self):
        super().__init__()  # "Add" aggregation (Step 5).
        self.conv = GCNConv(dataset.num_node_features, dataset.num_classes)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = self.conv(x, edge_index)

        return F.log_softmax(x, dim=1)
 
class Hypertrain:
    def __init__(self, args):      
        #cross entropy between predicted and actual labels
        self.loss_fn = nn.CrossEntropyLoss() #consider logits
        
    def train(self):
        return

    def eval(self, all_pred):
        return
        
def train(args):
    args.e = torch.zeros(args.ne, args.n_hidden).to(device)
    hypertrain = Hypertrain(args)
    pred_all, loss, test_err = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
    return test_err

    
def is_valid_dataset(parser, arg):
      if arg not in ["Cora", "Citeseer", "Pubmed"]:
           parser.error("This dataset is not a citation network!" % arg)
      else:
           return arg


def gen_synthetic_data(args, ne, nv):
    '''
    Generate synthetic data. 
    '''  
    #no replacement!
    args.n_hidden = 50
    n_labels = max(1, int(nv*.1))
    #numpy.random.choice(a, size=None, replace=True, p=None)
    args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)
    #args.label_idx = torch.zeros(n_labels, dtype=torch.int64).random_(0, nv) #torch.randint(torch.int64, nv, (int(nv*.1), ))
    args.labels = torch.ones(n_labels, dtype=torch.int64)
    args.labels[:n_labels//2] = 0
    args.n_cls = 2
    #labeled 
    args.vidx = torch.zeros((ne,), dtype=torch.int64).random_(0, nv-1) + 1 #np.random.randint(nv, (ne, 3))
    args.eidx = torch.zeros((nv,), dtype=torch.int64).random_(0, ne-1) + 1 #torch.random.randint(ne, (nv, 2))    
    args.v_weight = torch.ones((nv, 1)) / 2
    args.e_weight = torch.ones(ne, 1) / 3
    print(args)
    #train(args)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # Training settings
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', default="Cora", 
                        help='Citation Network. Choose one between (Cora, Citeseer, Pubmed)', type=lambda x: is_valid_dataset(parser, x))
    args = parser.parse_args()

    dataset = args.dataset
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(".", dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    gen_synthetic_data(args, ne=2, nv=3)


