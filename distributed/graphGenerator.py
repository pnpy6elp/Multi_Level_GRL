from torch_geometric.datasets import *
import networkx as nx
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx as fn

def load_graph(dataset):
    if dataset == "wikics":
        graph = WikiCS("/tmp/WikiCS")[0]
    elif dataset == "coauthor_physics":
        graph = Coauthor("/tmp/Physics", name="Physics")[0]
    elif dataset == "coauthor_cs":
        graph = Coauthor("/tmp/Coauthor_CS",name="CS")[0]
    elif dataset == "deezereu":
        graph =DeezerEurope("/tmp/DeezerEurope")[0]
    elif dataset == 'foursquare':
        file = 'dataset_WWW_friendship_new.txt'
        node_features = pd.read_csv("node_features_encoded.csv", index_col=0)
        node_features = node_features.iloc[:,1:]
        g = nx.read_edgelist(file , nodetype = int, edgetype='Freindship')
        g = fn(g)
        g["x"] = torch.tensor(node_features.values)
        graph = g
    return graph
