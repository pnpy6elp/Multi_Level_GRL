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
def get_features(g_nx): # get graph-based features
    i = fn(g_nx)
    features_li = []
    features = []
    vertex = i.x.shape[0]
    feat_dim = i.x.shape[1]
    edge = i.edge_index.shape[1]
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
    sc_dit["multilevel"] = ml_sc[0]
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
        elif md=="multilevel":
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