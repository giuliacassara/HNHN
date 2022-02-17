from ast import arg
import string
import graph
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import time
import numpy as np
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T


def is_valid_dataset(parser, arg):
      if arg not in ["Cora", "Citeseer", "Pubmed"]:
           parser.error("This dataset is not a citation network!" % arg)
      else:
           return arg

parser = argparse.ArgumentParser()

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

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__=='__main__':

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(path, args.dataset, transform=transform)
    data = dataset[0]
    data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = graph.GCN(dataset.num_features, 16, dataset.num_classes, data).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Init time
    t_total = time.time()
    best_val_acc = test_acc = 0
    for epoch in range(0, args.epochs):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
            f'Final Test: {test_acc:.4f}')
    print("Time: {:.4f}s".format(time.time() - t_total))
