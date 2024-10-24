# ML-GRL in the distributed environment
An implementation of ML-GRL.


# Run

## Environment Setup
You can either use Sockets for communication or utilize NFS. In the case of NFS, please follow the manual below.
1. Setup NFS.
- Master Node
```
sudo apt-get install nfs-kernel-server
```
- Worker Node
```
sudo apt-get install nfs-common
```

2. Run main.py

```
python main.py --num_epochs 10 --batch_size 128 --dataset wikics --model graphsage  --model_path ./new_model2 --core 35 --minor 100 --sim 0.95 --task link
--ip_config ip_config.txt
```
- `--num_epochs`: the number of epochs.
- `--batch_size`: batch size.
- `--dataset`: the name of dataset (wikics, coauthor_physics, coauthor_cs, deezereu, foursquare).
- `--model`: the model of graph representation learning (graphsage, arma, asap, gat, gt).
- `--model_path`: the path of the prediction models.
- `--partitioning`: Community detection algorithm (fastgreedy, label_propagatio, infomap, leiden, louvain).
- `--core`: the number of CPU cores for multiprocessing.
- `--minor`: the threshold of minor communities.
- `--sim`: the threshold of similarity.
- `--task`: the type of downstream task (link, node).
- `--ip_config`: IP configuration file.
