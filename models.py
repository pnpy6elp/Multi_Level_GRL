import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx as fn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GATConv, ASAPooling, LEConv, ARMAConv, TransformerConv
from torch_geometric.datasets import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from torch import Tensor
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
import networkx as nx
from networkx.algorithms import approximation
import igraph as ig
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, train_test_split_edges
import pandas as pd


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        
        self.conv1 = SAGEConv(input_dim, hidden_channels,dropout=0.5)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels,dropout=0.5)
        self.conv3 = torch.nn.Linear(hidden_channels, 64) 
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
class ARMA(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        
        self.conv1 = ARMAConv(input_dim, hidden_channels,dropout=0.5)
        self.conv2 = ARMAConv(hidden_channels, hidden_channels,dropout=0.5)
        self.conv3 = torch.nn.Linear(hidden_channels, 64) 
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
class ASAP(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()

        # pool = ASAPooling(input_dim, ratio=0.5, GNN=LEConv,
                          add_self_loops=False)
        self.conv1 = LEConv(input_dim, hidden_channels)
        self.conv2 = LEConv(hidden_channels, hidden_channels)
        self.conv3 = torch.nn.Linear(hidden_channels, 64) 
        
    def encode(self, x, edge_index):
        # x = self.pool(x, edge_index)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge
    

    class GT(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super(GT, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_channels,dropout=0.5) 
        self.conv2 = TransformerConv(hidden_channels, hidden_channels,dropout=0.5) #
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# building model -GAT
class GAT(torch.nn.Module): 
    def __init__(self, input_dim, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_channels,dropout=0.5) 
        self.conv2 = GATConv(hidden_channels, hidden_channels,dropout=0.5) 
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
def run_for_node(graph, epoch, batch, model_name):
    if model_name == "gat":
        model = GAT(graph.x.shape[1],256) 
    elif model_name == "gt":
        model = GT(graph.x.shape[1],256) 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    num_epochs = epoch
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph.x.to(torch.float), graph.edge_index)
        loss = loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()

    embedding = model(graph.x.to(torch.float),graph.edge_index)
    embedding = embedding.detach()
    return embedding 


def run_for_link(graph, n_epochs, batch, model_name):
    if model_name == "graphsage":
        model = GraphSAGE(graph.x.shape[1],batch)
    elif model_name == "arma":
        model = ARMA(graph.x.shape[1],batch)
    elif model_name == "asap":
        model = ASAP(graph.x.shape[1],batch)
    graph.x = graph.x.to(torch.float)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    split = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0.0,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
    )

    train_data, val_data, _ = split(graph)


    
    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor_for_GRL(model, val_data)

            
    embedding = model.encode(graph.x, graph.edge_index).detach()
    return embedding


@torch.no_grad()
def eval_link_predictor_for_GRL(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()


    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
