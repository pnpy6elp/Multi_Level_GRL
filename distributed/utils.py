import pandas
from igraph import *
from networkx.readwrite import json_graph
import json
import time
import pickle
from tqdm import tqdm
import numpy as np
from torch_geometric.utils.convert import from_networkx as fn
import networkx as nx
from networkx.algorithms import approximation
import time
import igraph as ig
from pssh.clients import ParallelSSHClient
import socket
import threading
from collections import defaultdict
import os

def handle(client):
    while True:
        try:
            message = client.recv(512*512)
            broadcast(message)

        except:
            client.close()
            break

def receive():
    print("receive....")
    timeout = 0
    while True:
        client, address = server.accept()
        print("Connected with {}".format(str(address)))
        data = b""
        while True:
            packet = client.recv(512*512)
            if not packet: break
            data += packet
        print("hihi")
        dit[address[0]] = pickle.loads(data)
        print("good")
        client.send('Connected to server!'.encode('ascii'))
        thread = threading.Thread(target=handle, args=(client,))
        thread.start()
        timeout += 1
        if timeout==len(hosts):
            break
def send(data, hosts):
    print("start")
    timeout = 0
    while True:
        client, address = server.accept()
        print("Connected with {}".format(str(address)))
        t1 = time.time()
        data_string = pickle.dumps(data[timeout])
        t2 = time.time() - t1
        print(f"done!! {t2} secs")
        print("Sending to {}".format(str(address)))
        t3 = time.time()
        client.send(data_string)
        client.close()
        t4 = time.time() - t3
        print("Transfer : {} secs".format(str(t4)))
        timeout += 1
        if timeout==len(hosts):
            break

def bubble_sort(array):
    n = len(array)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if array[j].vcount() < array[j + 1].vcount():
                array[j], array[j + 1] = array[j + 1], array[j]
    return array
def bubble_sort_reverse(array):
    n = len(array)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if array[j].vcount() > array[j + 1].vcount():
                array[j], array[j + 1] = array[j + 1], array[j]
    return array

def distribute_community(graph, num_clusters):
    boxes = {}
    graphs = {}
    for i in range(num_clusters):
        boxes[i] = 0
        graphs[i] = []
        
    graph = bubble_sort(graph)
    
    for i in graph:
        min_weight_box = min(boxes, key=boxes.get)
        boxes[min_weight_box] += i.vcount()
        graphs[min_weight_box].append(i)

    try:
        if not os.path.exists("partition"):
            os.makedirs("partition")
    except OSError:
        print("Error: Failed to create the directory.")
        
    for key, value in graphs.items():
        # save
        with open(f'partition/data{key}.pickle', 'wb') as f:
            pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
        print(f"======{key}======")
        for j in value:
            print("{vc} nodes, {ec} edges ".format(vc = j.vcount(), ec=j.ecount()))
    return graphs

# +
def get_membership():
    membership = {}
    cnt = 0
    for file in [f for f in os.listdir("./") if f.startswith("membership")]:
        with open('./'+file, 'rb') as f:
            data = pickle.load(f)
        for dt in data: # node list per subgraph
            membership[dt] = cnt
        cnt += 1
    membership = dict(sorted(membership.items()))
    membership= list(membership.values())
    return membership


def get_major_embed(embed_dit):
    for file in [f for f in os.listdir("./") if f.startswith("embed")]:
        with open('./'+file, 'rb') as f:
            data = pickle.load(f)
        embed_dit.update(data)
    return embed_dit


# -

def get_features(g_nx): # get graph-based features
    features_li = []
    features = []
    vertex = g_nx.number_of_nodes()
    feat_dim = len(next(iter(nx.get_node_attributes(g_nx, "x").values())))
    edge = g_nx.number_of_edges()
    dense = nx.density(g_nx)
    cen = nx.degree_centrality(g_nx)
    c_mean = np.array(list(cen.values())).mean()
    c_max = np.max(np.array(list(cen.values())))
    c_min = np.min(np.array(list(cen.values())))
    c_std = np.std(np.array(list(cen.values())))
    cluster_coeffi = approximation.average_clustering(g_nx)
    degrees = [val for (node, val) in g_nx.degree()]
    degree_min = min(degrees)
    degree_max = max(degrees)
    degree_mean = np.mean(np.array(degrees))
    degree_std = np.std(np.array(degrees))
    features.append(vertex)
    features.append(edge)
    features.append(feat_dim)
    features.append(dense)
    features.append(c_mean)
    features.append(c_min)
    features.append(c_max)
    features.append(c_std)
    features.append(cluster_coeffi)
    features.append(degree_mean)
    features.append(degree_max)
    features.append(degree_min)
    features.append(degree_std)
    features_li.append(features)
    feat_arr = np.array(features_li)
    return feat_arr
def predict_acc(feature): # to btain predicted accuracy
    sc_dit = {}
    fg_sc = fg_rf.predict(feature)
    im_sc = im_rf.predict(feature)
    lp_sc = lp_rf.predict(feature)
    ld_sc = ld_rf.predict(feature)
    ml_sc = ml_rf.predict(feature)
    nop_sc = nop_rf.predict(feature)
    sc_dit["fastgreedy"] = fg_sc[0]
    sc_dit["infomap"] = im_sc[0]
    sc_dit["label_propagation"] = lp_sc[0]
    sc_dit["leiden"] = ld_sc[0]
    sc_dit["louvain"] = ml_sc[0]
    sc_dit["no_partition"] = nop_sc[0]
    sc_dit = dict(sorted(sc_dit.items(), key=lambda x: x[1], reverse=True)) # reverse sort
    md = next(iter(sc_dit))
    score = sc_dit[md]
    print(f"Selected Algorithm : {md} ({score})")
    return sc_dit, score, md


def partitioning(g_nx, md): 
    status = 1
   
    if md != "no_partition":
        g_ig = ig.Graph.from_networkx(g_nx)
        g_ig = g_ig.as_undirected()
        if md=="fastgreedy":
            time1 = time.time()
            partitions = g_ig.community_fastgreedy()
            time2 = time.time() - time1
            partitions = partitions.as_clustering()
        elif md=="label_propagation":
            time1 = time.time()
            partitions = g_ig.community_label_propagation()
            time2 = time.time() - time1
        elif md=="infomap":
            time1 = time.time()
            partitions = g_ig.community_infomap()
            time2 = time.time() - time1
        elif md=="leiden":
            time1 = time.time()
            partitions = g_ig.community_leiden("modularity")
            time2 = time.time() - time1
        elif md=="louvain":
            time1 = time.time()
            partitions = g_ig.community_multilevel()
            time2 = time.time() - time1
        subsub = partitions.subgraphs()
    else: # no partitioning is selected
        time2 = 0.0
        g_ig = ig.Graph.from_networkx(g_nx)
        subsub = []
        subsub.append(g_ig)
        status = 0
    print(f"{md}'s time : {time2} secs")
    return subsub, time2, status


def set_attr_minor_super(ver_seq,clust_major,major_attr):
    res_li = []
    for i in clust_major.subgraphs()[max(major_attr)+1:]:
        vs11 = np.array(i.vs["x"])
        res = np.mean(vs11, axis=0)
        res_li.append(res)
    for i in range(len(major_attr)):
        ver_seq.vs[i]['x'] = major_attr[i].flatten()
    for i in range(len(clust_major.subgraphs()[max(major_attr)+1:])):
        ver_seq.vs[max(major_attr)+1+i]['x'] = res_li[i]
    return ver_seq
