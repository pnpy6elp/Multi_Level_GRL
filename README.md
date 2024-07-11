# Multi-Level Graph Representation Learning Through Predictive Community-based Partitioning
An implementation Multi-Level GRL.

# Abstract
> Graph representation learning (GRL) aims to map a graph into a low-dimensional vector space while preserving graph topology and node properties. As the size and complexity of graphs grow, existing GRL methods face scalability issues, and attempts have been made to address the problem through graph partitioning. However, these methods prioritize load balancing and communication costs over the semantic information of the graph, leading to performance degradation. In this study, we propose a novel GRL model, MultiLevel GRL (simply, ML-GRL), that recursively partitions the graph through the most appropriate community detection algorithm at each graph or partitioned subgraphs. To this end, we design a prediction model to predict the GRL accuracy of each community detection algorithm without partitioning the graph. We pre-train a prediction model by sampling a training dataset to widely cover various real-world graphs and extracting graph-based structural features. Based on this model, we dynamically select the community detection algorithm with the highest expected accuracy. ML-GRL improves learning accuracy by selecting the most effective subgraph across diverse community detection algorithms while improving the learning efficiency due to the parallel processing of partitioned subgraphs. Through extensive experiments on five popular graph datasets with two graph downstream tasks, we demonstrate the superiority of ML-GRL compared to the five representative GRL models regarding both learning accuracy and efficiency.

# Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

# Run
```
python main.py --num_epochs 10 --batch_size 128 --dataset wikics --model graphsage --model_path ./new_model2 --core 35 --task link
```
- `--num_epochs`: the number of epochs.
- `--batch_size`: batch size.
- `--dataset`: the name of dataset. (wikics, coauthor_physics, coauthor_cs, deezereu, foursquare)
- `--model`: the model of graph representation learning. (graphsage, arma, asap, gat, gt)
- `--model_path`: the path of the prediction models.
- `--core`: the number of CPU cores for multiprocessing.
- `--task`: the type of downstream task. (link, node)
