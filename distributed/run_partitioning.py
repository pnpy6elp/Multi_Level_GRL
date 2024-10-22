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
    parser.add_argument("--minor", type=int, default=100)
    parser.add_argument("--similarity", type=float, default=0.95)
    parser.add_argument("--model", type=str, default="graphsage")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--core", type=int, default=35)
    parser.add_argument("--host_num", type=int, default=0)
    
    args = parser.parse_args()
    print(f"Arguments: {args}")

    cmd = ""
    epoch = args.num_epochs
    batch = args.batch_size
    model_name = args.model
    prediction_path = args.model_path
    minor_thres = args.minor
    delta = args.delta
    core =args.core
    host_num =args.host_num
    
    current_depth = 1
    epochs = epoch
    batches = batch
    proc_num = 0
    start = time.time()
    
    with open(f'partition/data{host_num}.pickle', 'rb') as f:
        mjmj = pickle.load(f)
            
            
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
    
    data_string = pickle.dumps("done")
    ##time.sleep(60)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('master', PORT))
    client.send(data_string)

    with open('./membership.pickle', 'wb') as f:
        pickle.dump(membership, f, pickle.HIGHEST_PROTOCOL)

    

    
