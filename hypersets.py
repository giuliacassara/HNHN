
'''
using hypergraph representations for document classification.
'''
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
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Dataset

from data_helpers import adjacency_to_edge_index, incidence_to_adjacency, generate_incidence_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Hypergraph():
	def __init__(self, nv, ne, v_weight, e_weight, incidence_matrix) -> None:
		self.nv = nv
		self.ne = ne
		self.v_weight = v_weight
		self.e_weight = e_weight
		self.incidence_matrix = incidence_matrix


class InvariantModel(nn.Module):
	'''
	The class returns a hyperedge-level representation of an input hypergraph.
	Using the sum-pooling layer results in the invariance of the architecture to
	node permutation inside each hyperedge. This vector representation is then used for
	prediction, adding a GCN or GAT layer to predict each node's label.

	in_channels (int): Size of each input sample, or :obj:`-1` to derive
	the size from the first input(s) to the forward method.
	out_channels (int): Size of each output sample.

					Shapes:
					- input
									node features
									hyperedge indices
									hyperedge weights (optional)
					- output
									node features (reduced dimension)
	'''

	def __init__(self, in_channels, out_channels):
		super().__init__()  # "Sum" aggregation.
		self.phi = Seq(Linear(in_channels, out_channels),
					   ReLU(),
					   Linear(out_channels, out_channels))
		self.rho = torch.nn.Sigmoid()

	def forward(self, x):
		# compute the representation for each data point
		x = self.phi.forward(x)
		# sum up the representations
		# here I have assumed that x is 2D and the each row is representation of an input, so the following operation
		# will reduce the number of rows to 1, but it will keep the tensor as a
		# 2D tensor.
		x = torch.sum(x, dim=0, keepdim=True)
		# compute the output
		out = self.rho.forward(x)

		return out


class HyperEdgeConv(torch.nn.Module):
	def __init__(self, num_features, hidden_channels, num_classes):
		super().__init__()
		torch.manual_seed(1234567)
		# self.conv = GCNConv(dataset.num_node_features, dataset.num_classes)
		self.conv1 = GCNConv(num_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, num_classes)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=0.5, training=True)
		x = self.conv2(x, edge_index)
		return x
		# return F.log_softmax(x, dim=1)


class Hypertrain:
	def __init__(self, args):
		# cross entropy between predicted and actual labels
		self.loss_fn = nn.CrossEntropyLoss()  # consider logits

	def train(self):
		return

	def eval(self, all_pred):
		return


def train(data, optimizer, model):
	model.train()
	optimizer.zero_grad()
	out = model(data.x, data.adj_t)
	loss = F.nll_loss(out, data.y)
	loss.backward()
	optimizer.step()
	return float(loss)


def is_valid_dataset(parser, arg):
	if arg not in ["Cora", "Citeseer", "Pubmed"]:
		parser.error("This dataset is not a citation network!" % arg)
	else:
		return arg


def test_GCN(num_features, hidden_channels, num_classes, data):
	model = HyperEdgeConv(num_features, hidden_channels, num_classes)
	print(model)
	h = model(data.x, data.edge_index)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	for epoch in range(1, 201):
		loss = train(data)



def gen_synthetic_data(args, ne, nv):
	'''
	Generate synthetic data. 
	'''
	n_labels = max(1, int(nv*.1))
	label_idx = torch.from_numpy(np.random.choice(
		nv, size=(n_labels,), replace=False)).to(torch.int64)
	labels = torch.ones(n_labels, dtype=torch.int64)
	labels[:n_labels//2] = 0
	n_cls = 2
	p = 0.6
	#incidence_matrix = torch.randint(0, 2, (nv, ne))
	incidence_matrix = generate_incidence_matrix(nv, ne, 3, p)
	print(incidence_matrix)
	A = incidence_to_adjacency(incidence_matrix.T)
	print("adjacency matrix from incidence ", A)
	print("edge index from adjacency matrix ", adjacency_to_edge_index(A))
	feature_dimension = 10
	initial_node_features = torch.randn(nv, feature_dimension)
	# for every hyperedge inside the incidence matrix:
	print("These are the initial node features: ", initial_node_features)
	edge_index = adjacency_to_edge_index(A)
	hyperedge_feature = torch.empty(size=(ne, feature_dimension))

	for edge_iter in range(0, ne):
		hyperedge_tensor = incidence_matrix[:, edge_iter]
		# find positions of 1 in tensor
		condition = hyperedge_tensor > 0
		indices = condition.nonzero()
		# print(indices)
		# stack the node features in a 2d tensor
		single_hyperedge_feature = torch.empty(
			size=(len(indices), feature_dimension))
		for count, v_index in enumerate(indices):
			single_hyperedge_feature[count] = initial_node_features[v_index]
			# hyperedge_feature.vstack(nf)
		# compute the readout = deepset()
		print("This tensor is a collection of all features in a hyperedge before DeepSet: ",
			  single_hyperedge_feature)
		deepset = InvariantModel(feature_dimension, feature_dimension)
		readout = deepset.forward(single_hyperedge_feature)
		hyperedge_feature[edge_iter] = readout
	
		# build the Data object
	data = Data(x=hyperedge_feature, edge_index=edge_index)
	#build data y
	print(data)
	# device = torch.device('cuda')
	# data = data.to(device)
	# compute the line graph, get the adjacency matrix, than the edge index

	test_GCN(feature_dimension, int(feature_dimension/2), n_cls, data)
	# Readout is the new feature vector representing the hyperedge. Now


if __name__ == '__main__':
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
	# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
	dataset = Planetoid(".", dataset, transform=T.NormalizeFeatures())
	data = dataset[0]

	gen_synthetic_data(args, ne=4, nv=5)
