import multiprocessing
import argparse
import time
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import *
from torch_geometric.utils.convert import from_networkx as fn
import argparse
import networkx as nx
import igraph as ig
import joblib
import pandas as pd
import math
from .ML_GRL import *
from .utils import *
from .tasks import *
from .models import *

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser("Multi-Level GRL")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--minor", type=int, default=100)
    parser.add_argument("--similarity", type=float, default=0.95)
    parser.add_argument("--dataset", type=str, default="wikics")
    parser.add_argument("--model", type=str, default="graphsage")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--task", type=str, default="link")
    parser.add_argument("--core", type=int, default=35)
    
    args = parser.parse_args()

    dataset = args.dataset
    epoch = args.num_epochs
    batch = args.batch_size
    model_name = args.model
    prediction_path = args.model_path
    minor_thres = args.minor
    delta = args.delta
    core =args.core
    task = args.task


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


    global fg_rf
    global im_rf
    global lp_rf
    global ld_rf
    global ml_rf
        
    fg_rf = joblib.load(f"{prediction_path}/fastgreedy.pkl")
    im_rf = joblib.load(f"{prediction_path}/infomap.pkl")
    lp_rf = joblib.load(f"{prediction_path}/label.pkl")
    ld_rf = joblib.load(f"{prediction_path}/leiden.pkl")
    ml_rf = joblib.load(f"{prediction_path}/louvain.pkl")
    nop_rf = joblib.load(f"{prediction_path}/nopartition.pkl")


    origin_x = graph
    if dataset == 'foursquare':
        g_nx0 = to_networkx(origin_x,node_attrs=["x"])
    else:
        g_nx0 = to_networkx(origin_x,node_attrs=["x","y"])
    g_nx0 = g_nx0.to_undirected()
    feat = get_features(g_nx0)
    sc_dit, score, first_md = predict_acc(feat)
    print(f"prediction result : {first_md} & {score}")
    first_sub, first_time, status = partitioning(g_nx0, first_md,minor_thres)
    print(f"1st CD Time : {first_time} secs")

    manager = multiprocessing.Manager()
    degree = manager.list()
    subgraphs_final = manager.list()
    
    epochs = epoch
    batches = batch
    mjmj = [x for x in first_sub if x.vcount() >= minor_thres] # major community
    mnmn = [x for x in first_sub if x.vcount() < minor_thres] # minor community
    tmp = math.ceil(len(mjmj)/core) # the number of graph / the number of cores

    
    current_depth = 1
    epochs = epoch
    batches = batch
    proc_num = 0
    start = time.time()
    for i in range(tmp):
        graph_lili = mjmj[i*core:(i+1)*core]
        jobs = []
        for graph in graph_lili:
            graph = graph.to_networkx()
            p = multiprocessing.Process(target=recursive_partitioning,args=[graph, minor_thres,delta, current_depth, subgraphs_final])
            proc_num += 1
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    end = time.time() - start
    print(f"Overall partitioning time : {first_time+end} secs")
    
        
    subgraphs_final = list(subgraphs_final)
    subgraphs_final.extend(mnmn)
    subgraphs = list(subgraphs_final)


    minor_node_id = []
    major_node_id = []
    for i in subgraphs:
        if(i.vcount() < minor_thres):
            minor_node_id.extend(i.vs["_nx_name"])
        else:
            major_node_id.extend(i.vs["_nx_name"])
            
    # global graph construction
    major_super_feat, membership_dit = merge_major(first_sub, minor_thres, minor_node_id)
    
    # reassign minor's membership
    start = max(major_super_feat.keys()) + 1
    for i in minor_node_id:
        membership_dit[i] = start
        start += 1
    ig_origin = ig.Graph.from_networkx(g_nx0)
    membership_dit = dict(sorted(membership_dit.items()))
    membership = list(membership_dit.values())
    clust = ig.VertexClustering(ig_origin, membership=membership)
    
    Global = ig_origin.copy()
    Global.contract_vertices(membership, combine_attrs="first")
    Global.simplify(combine_edges="ignore")
    print(Global.summary())
    for i in range(len(major_super_feat)):
        Global.vs[i]['x'] = major_super_feat[i].flatten()

    superG = Global.copy()
    
    minor_com =[]
    for i in subgraphs:
        if i.vcount() < minor_thres:
            minor_com.append(i)
    minor_ig = superG.copy()
    deleted = []
    for v in range(minor_ig.vcount()):
        if minor_ig.vs['_nx_name'][v] in major_node_id: 
            deleted.append(v)
    minor_ig.delete_vertices(deleted)
    
    subgraphs_sorted = bubble_sort_reverse(subgraphs)
    memdit_for_minor = {}
    start = 0
    memsh = {}
    aa = 0
    for i in subgraphs_sorted:
        for vv in i.vs["_nx_name"]:
            memsh[vv] = aa
            memdit_for_minor[vv] = start
        aa += 1
        start += 1
        
    memdit_for_minor_dit = dict(sorted(memdit_for_minor.items()))
    memdit_for_minor = list(memdit_for_minor_dit.values())
    Global_minor = ig_origin.copy()
    Global_minor.contract_vertices(memdit_for_minor, combine_attrs="first")
    Global_minor.simplify(combine_edges="ignore")

    from collections import Counter
    mj_mn_counter = Counter(memdit_for_minor)

    Global_minor.vs["num"] = list(dict(sorted(mj_mn_counter.items())).values())
    
    merge_t1 = time.time()
    merged_Global = process_graph(Global_minor, minor_thres, minor_node_id, "link")
    merge_t2 = time.time() - merge_t1
    
    
    isolated = []
    memdit_for_minor2 = memdit_for_minor_dit.copy()
    final_member_dit = {}
    memdit_tmp = memdit_for_minor2.copy()
    start = 0
    aa = 0
    for i in merged_Global.vs:
        if len(merged_Global.vs[i.index]['nx_list']) > 1:
            all_nodes = []
            nx_li = merged_Global.vs[i.index]['nx_list']
            for j in nx_li:
                all_nodes.extend([k for k, v in memdit_for_minor2.items() if v == memdit_tmp[j]])
            for n_id in all_nodes:
                final_member_dit[n_id] = start
            aa += len(all_nodes)
            start += 1
        else:
            one_id = merged_Global.vs[i.index]['nx_list'][0]
            if one_id in major_node_id:
                all_nodes = []
                all_nodes.extend([k for k, v in memdit_for_minor2.items() if v == memdit_tmp[one_id]])
                for n_id in all_nodes:
                    final_member_dit[n_id] = start
                aa += len(all_nodes)
                start += 1
            else:
                all_nodes = []
                all_nodes.extend([k for k, v in memdit_for_minor2.items() if v == memdit_tmp[one_id]])
                isolated.extend(all_nodes)
                
    isolated_nodes = []
    for i_node in isolated:
        isolated_nodes.extend([k for k, v in memdit_for_minor2.items() if v == memdit_for_minor2[i_node]])
        
    for n_id in isolated_nodes:
        final_member_dit[n_id] = start
    final_membership_dit = dict(sorted(final_member_dit.items()))
    final_membership= list(final_membership_dit.values())

    clust_major = ig.VertexClustering(ig_origin, membership=final_membership)
    Global_final = ig_origin.copy() # Global for major grl
    Global_final.contract_vertices(final_membership, combine_attrs="first")
    Global_final.simplify(combine_edges="ignore")
    Global_final = set_attr_minor_super(Global_final,clust_major,major_super_feat)
    print(Global_final.summary())

    from tqdm import tqdm
    
    import math
    major_subgraphs = [s for s in subgraphs if s.vcount() >= minor_thres]
    
    tmp = math.ceil(len(major_subgraphs) / core)
    
    
    manager = multiprocessing.Manager()
    embed_dit = manager.dict()
    args_list = []
    start = time.time()
    
    start = time.time()
    for i in range(tmp):
        graph_lili = major_subgraphs[i*core:(i+1)*core]
        jobs = []
        for graph in graph_lili:
            sub_g = graph
            args_li = (sub_g,clust_major,ig_origin,Global_final,model_name, embed_dit, task)
            p = multiprocessing.Process(target=major_grl,args=[args_li])
            proc_num += 1
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    
    end = time.time() - start

    superG_nx = superG.to_networkx()
    global_id_li = superG.vs["_nx_name"]
    superG_pyg = fn(superG_nx)
    tt = time.time()

    if task=="link":
        minor_result = run_for_link(superG_pyg, epoch, batch,model_name)
    elif task=="link":
        minor_result = run_for_node(superG_pyg, epoch, batch,model_name)
    tt2 = time.time() - tt
    print(f"minor grl time : {tt2} secs")
    for num in range(len(minor_node_id)):
        embed_dit[minor_node_id[num]] = minor_result[-len(minor_node_id):][num]
    tmp_emb = list(dict(sorted(embed_dit.items())).values())
    tmp_emb = torch.stack(tmp_emb)

    
    tmp_emb = list(dict(sorted(embed_dit.items())).values())
    tmp_emb = torch.stack(tmp_emb)
    
    origin_x["embedding"] = tmp_emb

    if task=="link":
        acc = run_prediction(origin_x, epoch)
    elif task=="link":
        acc = run_classification(tmp_emb,origin_x.y,128,10)

    print(acc)







