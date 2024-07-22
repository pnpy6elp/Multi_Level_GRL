# Multi-Level Graph Representation Learning Through Predictive Community-based Partitioning
An implementation of ML-GRL.

# Abstract
> Graph representation learning (GRL) aims to map a graph into a low-dimensional vector space while preserving graph topology and node properties. This study proposes a novel GRL model, Multi-Level GRL (simply, ML-GRL), that recursively partitions input graphs by selecting the most appropriate community detection algorithm at each graph or partitioned subgraph. To preserve the relationship between subgraphs, ML-GRL incorporates global graphs that effectively maintain the overall topology. ML-GRL employs a prediction model, which is pre-trained using graph-based features and covers a wide range of graph distributions, to estimate GRL accuracy of each community detection algorithm without partitioning graphs or subgraphs and evaluating them. ML-GRL improves learning accuracy by selecting the most effective community detection algorithm while enhancing learning efficiency from parallel processing of partitioned subgraphs. Through extensive experiments with two different tasks, we demonstrate ML-GRL’s superiority over the six representative GRL models in terms of both learning accuracy and efficiency. Specifically, ML-GRL not only improves the accuracy of existing GRL models by 3.68 ∼ 47.59% for link prediction and 1.75 ∼ 40.90% for node classification but also significantly reduces their running time by 9.63 ∼ 62.71% and 7.14 ∼ 82.14%, respectively. 


# Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

# Dataset
Dataset used in the experiment.
- [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

Other datasets can be obtained from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) library. 

# GRL Model
- [ClusterSCL](https://github.com/wyl7/ClusterSCL/tree/main) (WWW 2022)

Other models are from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) library. 

# Run
## Offline Phase
```
python run_offline.py --num_epochs 10 --batch_size 128 --dataset wikics --model graphsage --model_path ./new_model2 --core 35 --minor 100 --sim 0.95 --task link
```
- `--num_epochs`: the number of epochs.
- `--batch_size`: batch size.
- `--dataset`: the name of dataset (wikics, coauthor_physics, coauthor_cs, deezereu, foursquare).
- `--model`: the model of graph representation learning (graphsage, arma, asap, gat, gt).
- `--model_path`: the path of the prediction models.
- `--core`: the number of CPU cores for multiprocessing.
- `--minor`: the threshold of minor communities.
- `--sim`: the threshold of similarity.
- `--task`: the type of downstream task (link, node).

## Online phase
```
python run_online.py --num_neighbors 5 --size 128 --partitioning leiden --model graphsage --task link
```
- `--num_neighbors`: the number of neighbors.
- `--size`: the number of fixed-sampled nodes.
- `--partitioning`: Community detection algorithm (fastgreedy, label_propagatio, infomap, leiden, louvain).
- `--model`: the model of graph representation learning (graphsage, arma, asap, gat, gt).
- `--task`: the type of downstream task (link, node).
