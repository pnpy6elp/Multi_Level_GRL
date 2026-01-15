from torch_geometric.datasets import *
from torch_geometric.loader import NeighborLoader
import argparse
from utils import get_features, partitioning
from torch_geometric.utils.convert import to_networkx
import numpy as np
from tasks import *
from models import *
import copy
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def neighbor_sampling(graph, num_neighbor, size):
    sample_loader = NeighborLoader(
    graph,
    # Sample n neighbors for each node for depth k
    num_neighbors=[num_neighbor] * 1,
    # Use a batch size for sampling training nodes -> the number of fixed-sampled nodes
    batch_size=size,
    )
    result = []
    for i in sample_loader:
        result.append(i)

    return result

def MAPE(y_test, y_pred):
    tmp = np.abs((y_test - y_pred) / y_test)
    tmp = tmp[np.isfinite(tmp)]
    return np.mean(tmp) * 100 

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Offline Phase of ML-GRL")

    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--partitioning", type=str, default="leiden")
    parser.add_argument("--task", type=str, default="link")
    parser.add_argument("--model", type=str, default="graphsage")


    args = parser.parse_args()

    num_neighbor = args.num_neighbors
    size = args.size
    algorithm = args.partitioning
    task = args.task
    model_name = args.model

    # Cora, CiteSeer, PubMed, u, Amazon, Reddit, Facebook Page
    cora = Planetoid("./Cora","Cora")[0]
    citeseer = Planetoid("./CiteSeer","CiteSeer")[0]
    pubmed = Planetoid("./PubMed","PubMed")[0]
    flickr = Flickr("./Flickr")[0]
    amazon = Amazon("./Amazon","Computers")[0]
    reddit = Reddit2("./Reddit2")[0]
    #facebook = FacebookPagePage("./FacebookPagePage")[0]


    graph_list = []
    graph_list.append(cora)
    graph_list.append(citeseer)
    graph_list.append(pubmed)
    graph_list.append(flickr)
    graph_list.append(amazon)
    graph_list.append(reddit)
    graph_list.append(facebook)

    sample_list = []

    for g in graph_list:
        sample_list.extend(neighbor_sampling(g, num_neighbor, size))
    
    features_all = []
    accuracy = []

    for sample in sample_list:
        gg = copy.copy(sample)
        sample_nx = to_networkx(sample,node_attrs=["x"])
        features = get_features(sample_nx)
        features_all.append(features)

        subgraphs, _, _ = partitioning(sample_nx, algorithm)

        id_li = []
        embedding_list = []

        for sub_g in subgraphs:
            sub_g = sub_g.to_networkx()
            sub_g = fn(sub_g)

            id_li = id_li + list(sub_g.nodes)

            if task=="link":
                result = run_for_link(sub_g, 10, 256,model_name)
            elif task=="link":
                result = run_for_node(sub_g, 10, 256,model_name)

            embedding_list.append(result)

        embed = torch.cat(embedding_list,0)
        
        embed_dit = {}
        for i in range(len(id_li)):
            embed_dit[id_li[i]] = embed[i]
        tmp_emb = list(dict(sorted(embed_dit.items())).values())
        tmp_emb = torch.stack(tmp_emb)
        gg["embedding"] = tmp_emb


        if task=="link":
            acc = run_prediction(gg, 10)
        elif task=="link":
            acc = run_classification(tmp_emb,gg.y,128,10)

        accuracy.append(acc)

    X = np.array(features_all)
    y = np.array(accuracy)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

    # You can use other ML-based models
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    result_lr = lr.predict(x_test)
    knn_cl = KNeighborsRegressor(n_neighbors=3)
    knn_cl.fit(x_train, y_train)
    result_knn = knn_cl.predict(x_test)
    dt_cl = DecisionTreeRegressor()
    dt_cl.fit(x_train, y_train)
    result_dt = dt_cl.predict(x_test)
    rf_cl = RandomForestRegressor()
    rf_cl.fit(x_train, y_train)
    result_rf = rf_cl.predict(x_test)

    mape_lr = MAPE(y_test,result_lr)
    mape_knn = MAPE(y_test,result_knn)
    mape_dt = MAPE(y_test,result_dt)
    mape_rf = MAPE(y_test,result_rf)

    mape_dit = {}
    mape_dit["lr"] = mape_lr
    mape_dit["knn"] =mape_knn
    mape_dit["dt"] = mape_dt
    mape_dit["rf"] = mape_rf
    
    mape_dit = dict(sorted(mape_dit.items(), key=lambda x: x[1], reverse=True)) # reverse sort
    md = next(iter(mape_dit))
    print(f"Best model : {md}\n MAPE : {mape_dit[md]}")
    if md == "lr":
        joblib.dump(lr, f"{algorithm}_{task}_{md}.pkl")
    elif md == "knn":
        joblib.dump(knn_cl, f"{algorithm}_{task}_{md}.pkl")
    elif md == "dt":
        joblib.dump(dt_cl, f"{algorithm}_{task}_{md}.pkl")
    elif md == "rf":
        joblib.dump(rf_cl, f"{algorithm}_{task}_{md}.pkl")

    








        

    
    
