import torch
import torch_geometric.transforms as T

from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T

from torch_geometric.datasets import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from torch_geometric.utils.convert import from_networkx as fn
from torch import Tensor
import copy
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pickle
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import approximation
import numpy as np
import igraph as ig
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


# downstream task : link prediction
class LinkClassifier(torch.nn.Module):
    def __init_(self):
        super(LinkClassifier, self).__init__()
        
    def forward(self, embedding, edge_label_index):
        return (embedding[edge_label_index[0]] * embedding[edge_label_index[1]]).sum(dim=-1)
    
def train_link_predictor(train_data,val_data,n_epochs=10):
    model = LinkClassifier()
    
    for epoch in range(1, n_epochs + 1):
        
        
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
        
        out = model(train_data.embedding, edge_label_index).view(-1)

        val_auc = eval_link_predictor(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Val AUC: {val_auc}")
    return model

@torch.no_grad()
def eval_link_predictor(model, test_data):

    model.eval()
    out = model(test_data.embedding, test_data.edge_label_index)

    return roc_auc_score(test_data.edge_label.cpu().numpy(), out.cpu().numpy())

class NodeClassifier(torch.nn.Module):
    def __init__(self,input_dim,classes):
        super(NodeClassifier, self).__init__()
        self.layers = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32,classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
from tqdm import trange
def run_classification(embed,label,batch_size,n_epochs):
    model = NodeClassifier(embed.shape[1],max(label.unique())+1) # +1 is when using flickr
    x_train, x_test, y_train, y_test = train_test_split(embed, label, test_size=0.2, shuffle=True, random_state=45) 

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=45)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    batches_per_epoch = len(x_train) // batch_size
    
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(x_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) == y_val).float().mean()
        ce = float(ce)
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        
    model.load_state_dict(best_weights)

    pred = model(x_test).argmax(dim=1)
    correct = (pred == y_test).sum()
    acc = int(correct) / int(x_test.shape[0])
    print(f"Test Accuracy : {acc}")
    return acc


def run_prediction(graph, epoch):
    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = split(graph)



    model = train_link_predictor(train_data, val_data, epoch)

    test_auc = eval_link_predictor(model, test_data)
    print(test_auc)
    return test_auc