from utils import ensure_dir 
from utils import get_timestamp
from utils import load_yaml
from utils import shell

from data import build_dataloader
from data import build_dataset
from data import data_preprocessing
from data import setup_mln_iteration
from data import read_txt_triples, read_triple

from models import pLogicNet

from pLogicNet.utils import cmd_mln

from pathlib import Path


def train(data, path="./record", dev=False, iterations=2, args=None,
          entity_to_idx=True, relation_to_idx=True):
    
    if args is None:
        args = load_yaml(Path(__file__).absolute().parent 
                         / "configs" / "default.yaml")
    
    dataspace = ensure_dir(path)
    if dev:
        timestamp = get_timestamp()
    else:
        timestamp = "test_timestamp"
        
    workspace = ensure_dir(dataspace / timestamp)
    data_preprocessing("./pLogicNet/mln/mln", data, dataspace, workspace)
    
    for iter_id in range(iterations):
        workspace_iter = setup_mln_iteration(dataspace, iter_id, workspace)
        triples = read_txt_triples(workspace_iter / "train_kge.txt",
                                   entity_to_idx, relation_to_idx)
        hidden_triples = read_triple(workspace_iter / "hidden.txt", 
                                     triples["entity2id"], triples["relation2id"])
        datasets = build_dataset(workspace_iter)
        dataloader = build_dataloader(datasets)
        if iter_id == 0:
            model = pLogicNet(triples["nentities"], triples["nrelations"],
                              args.model_name, args.hidden_dim, args.gamma, 
                              args.double_entity_embedding, 
                              args.double_relation_embedding)
            
            model.train(dataloader, hidden_triples, workspace_iter, args=args,
                        max_steps=args.max_steps, lr=args.lr, log_step=args.log_step)
            shell(cmd_mln(
                mln = Path(__file__).absolute().parent / "pLogicNet" / "mln" / "mln",
                main_path = dataspace.absolute(),
                workspace_path = workspace_iter.absolute(),
                preprocessing = False,
                mln_threads = args.mln_threads,
                mln_iters = args.mln_iters,
                mln_lr = args.mln_lr,
                mln_threshold_of_rule = args.mln_threshold_of_rule
            ))
        else:
            pass

    
if __name__ == "__main__":
    # data = "./data/neo4j/neo4j.txt"
    data = "/root/InferenceEngine/pLogicNet/data/FB15k/train.txt"
    record = "./record"
    import shutil
    
    # try:
    #     shutil.rmtree(record)
    # except FileNotFoundError:
    #     ensure_dir(record)
        
    train(data)