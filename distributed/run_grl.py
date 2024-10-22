from ..ML_GRL import *
from utils import *
from ..tasks import *
from ..models import *
import socket
import multiprocessing

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser("DistGRL")

    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="graphsage")
    parser.add_argument("--core", type=int, default=35)
    parser.add_argument("--host_num", type=int, default=0)
    
    args = parser.parse_args()
    print(f"Arguments: {args}")

    epoch = args.num_epochs
    batch = args.batch_size
    model_name = args.model
    core =args.core
    host_num =args.host_num
    
    with open(f'partition/data{host_num}.pickle', 'rb') as f:
        subgraphs = pickle.load(f)
        
    with open(f'result_dit.pickle', 'rb') as f:
        result_dit = pickle.load(f)
        
    major_subgraphs = [s for s in subgraphs if s.vcount() >= minor_thres]
    
    tmp = math.ceil(len(major_subgraphs) / core)
    
    clust_major = result_dit['clust_major']
    Global_final = result_dit['Global_final']
    
    
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
    
    data_string = pickle.dumps("done")
    ##time.sleep(60)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('master', PORT))
    client.send(data_string)
    
    
    
    with open(f'./embed{host_num}.pickle', 'wb') as f:
        pickle.dump(embed_dit, f, pickle.HIGHEST_PROTOCOL)
    