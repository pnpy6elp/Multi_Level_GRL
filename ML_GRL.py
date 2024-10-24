from utils import *
from models import *
import networkx as nx
import heapq

def recursive_partitioning(graph, minor_thres, delta, current_depth, subgraphs_final):
    
    if len(graph.nodes)<minor_thres:
        print("It is minor community. So, partitioning is stopped.")
        subgraphs_final.extend(subgraphs)
        return [graph]

    feat = get_features(graph)
    sc_dit, score, md = predict_acc(feat)

    for ss in sc_dit.keys():
        subgraphs, time, status = partitioning(graph, md, minor_thres)
        if status == 0:  # no partitioning
            subgraphs_final.extend(subgraphs)
            return subgraphs


        subgraphs_sorted = bubble_sort(subgraphs)

        
        if ((subgraphs_sorted[0].vcount()/len(graph.nodes)) < delta):
            break
        

    result_subgraphs = []

    for subgraph in subgraphs: # you can run this on parallel with multiprocessing
        current_depth = current_depth + 1
        subgraph = subgraph.to_networkx()
        result_subgraphs.extend(recursive_partitioning(subgraph, minor_thres, delta, current_depth,subgraphs_final))

    return result_subgraphs




# Merge two nodes in the graph, connecting neighbors
def merge_nodes(graph, nodes_to_merge, new_num, nx_list_feature, task):
    new_node = len(graph.nodes)
    
    # Add the merged node with attributes
    if task == "node":
        graph.add_node(new_node, _nx_name=graph.nodes[nodes_to_merge[0]]['_nx_name'],
                       x=graph.nodes[nodes_to_merge[0]]['x'],
                       y=graph.nodes[nodes_to_merge[0]]['y'],
                       num=new_num, nx_list=nx_list_feature)
    elif task == "link":
        graph.add_node(new_node, _nx_name=graph.nodes[nodes_to_merge[0]]['_nx_name'],
                       x=graph.nodes[nodes_to_merge[0]]['x'],
                       num=new_num, nx_list=nx_list_feature)
    
    # Add edges for neighbors not in nodes_to_merge
    edges_to_add = []
    for n in nodes_to_merge:
        neighbors = graph.neighbors(n)
        for neighbor in neighbors:
            if neighbor not in nodes_to_merge:
                edges_to_add.append((new_node, neighbor))
    
    graph.add_edges_from(edges_to_add)
    graph.remove_nodes_from(nodes_to_merge)

    return graph

# Process the graph to merge minor communities
def process_graph(graph, minor_thres, minor_node_id, task):
    # Initialize heap for nodes sorted by 'num'
    heap = [(graph.nodes[n]['num'], n) for n in graph.nodes if graph.degree[n] > 0]
    heapq.heapify(heap)

    # Cache neighbors for each node to avoid recalculating
    node_neighbors = {n: list(graph.neighbors(n)) for n in graph.nodes}

    while heap:
        # Extract the node with the smallest 'num'
        _, start_node = heapq.heappop(heap)
        
        if graph.nodes[start_node]['num'] >= minor_thres:
            break

        # Get the first 1-hop neighbor
        neighbors = node_neighbors[start_node]
        if not neighbors:
            continue
        
        neighbors_minor = neighbors[0]
        
        # Merge the nodes
        current = graph.nodes[start_node]['num'] + graph.nodes[neighbors_minor]['num']
        nx_list = graph.nodes[start_node]['nx_list'] + graph.nodes[neighbors_minor]['nx_list']
        
        graph = merge_nodes(graph, [start_node, neighbors_minor], current, nx_list, task)

        # Update the heap and neighbor cache after merging
        for n in [start_node, neighbors_minor]:
            if n in node_neighbors:
                del node_neighbors[n]
        
        # Update heap with the new node
        heapq.heappush(heap, (current, len(graph.nodes) - 1))
    
    return graph


def merge_major(subgraphs, minor_thres, minor_node_id):
    major_super_feat = {}
    membership_dit = {}
    start = 0
    aa = 0
    for i in subgraphs:
        if i.vcount() >= minor_thres:
            feat_mat = []
            if len([x for x in i.vs['_nx_name'] if x not in minor_node_id])>0:
                for n in range(i.vcount()):
                    if i.vs['_nx_name'][n] not in minor_node_id:
                        membership_dit[i.vs['_nx_name'][n]] = start
                        feat_mat.append(i.vs['x'][n])
                        aa+= 1
                major_super_feat[start] = np.mean(feat_mat,axis=0)
                start += 1
    return major_super_feat,membership_dit




def major_grl(args): 
    sub_g,clust,ig_origin,Global_final,model_name, embed_dit, task = args
    total_len = sub_g.vcount()
        
    tmp_member = (clust.membership).copy()
    com_n = tmp_member[sub_g.vs["_nx_name"][0]]
    
    same_cluster_nodes = [] # Other nodes in the same cluster (excluding nodes in the current subgraph)
    for m in range(len(tmp_member)): # all nodes in graph
        if tmp_member[m] > com_n: # tmp_member[m] -> cluster id, m (index) -> _nx_name 
            tmp_member[m] = tmp_member[m] + (total_len-1)
        elif (tmp_member[m] == com_n) and (m not in sub_g.vs["_nx_name"]):
            same_cluster_nodes.append(m)
            tmp_member[m] = max(clust.membership) + total_len
    

    new_major_id = com_n
    for i in sub_g.vs["_nx_name"]:
        tmp_member[i] = new_major_id
        new_major_id += 1
    clust1 = ig.VertexClustering(ig_origin, membership=tmp_member)
    nn = 0
    res_li = []

    graph_new = ig_origin.copy()
    graph_new.contract_vertices(tmp_member, combine_attrs="first")
    idx = 0
    for i in range(graph_new.vcount()):
        if i < com_n and i >= new_major_id:
            if idx != com_n:
                graph_new.vs['x'] = Global_final.vs['x'][idx]
            idx += 1
    graph_new.simplify(combine_edges="ignore")
    cnt = 0
    deleted = []
    for v in range(graph_new.vcount()):
        if graph_new.vs['_nx_name'][v] in same_cluster_nodes: 
            deleted.append(v)
            cnt += 1
    graph_new.delete_vertices(deleted)
    major_id_list = graph_new.vs["_nx_name"][com_n:com_n+total_len+1]


    graph_new = graph_new.to_networkx()


    graph_new = fn(graph_new)
    if task=="link":
        result = run_for_link(graph_new, 10, 256,model_name)
    elif task=="node":
        result = run_for_node(graph_new, 10, 256,model_name)
    else:
        raise Exception("It is not supported.")
    
    

    for num in range(len(major_id_list)):
        embed_dit[major_id_list[num]] = result[com_n:com_n+total_len+1][num]
    return (major_id_list, result, com_n, total_len)
