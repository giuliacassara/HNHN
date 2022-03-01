import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Dataset
from scipy.sparse import issparse, coo_matrix, dok_matrix, csr_matrix

def from_sparse_matrix_to_edgeindex(A):
	r"""Converts a scipy sparse matrix to edge indices and edge attributes.
	Args:
					A (scipy.sparse): A sparse matrix.
	"""
	A = A.tocoo()
	row = torch.from_numpy(A.row).to(torch.long)
	col = torch.from_numpy(A.col).to(torch.long)
	edge_index = torch.stack([row, col], dim=0)
	# edge_weight = torch.from_numpy(A.data)
	return edge_index


def adjacency_to_edge_index(A):
	# adj = A.coo()
	# row, col, edge_attr = adj.coo()
	# edge_index = torch.stack([row, col], dim=0)
	edge_index = (A > 0).nonzero().t()
	return edge_index


def incidence_to_adjacency(M, s=1, weights=False):
	# M = csr_matrix(M)
	A = torch.matmul(M, M.T)
	print(A)
	new = torch.where(A > 1, 1, A)
	adjacency = new.fill_diagonal_(0)
	return adjacency


def generate_incidence_matrix(n, d, k, p_drop):
	sample = torch.rand(n, d).topk(k, dim=1).indices
	mask = torch.zeros(n, d, dtype=torch.bool)
	mask.scatter_(dim=1, index=sample, value=True)
	float_tensor = mask.float()
	print(float_tensor)
	output = F.dropout(float_tensor, p_drop)
	output = output.long()
	output = torch.where(output > 1, 1, output) #pork-around
	return output