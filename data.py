import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from pLogicNet.utils import cmd_mln
from pLogicNet.kge.dataloader import TrainDataset, BidirectionalOneShotIterator

import shutil
from utils import ensure_dir, shell


def data_preprocessing(mln_path, data, dataspace="./record", workspace=None):
    dataspace = ensure_dir(dataspace)
    if workspace is None:
        workspace = dataspace
        
    shutil.copy(data, dataspace / "train.txt")
    shutil.copy(data, dataspace / "train_augmented.txt")
    
    shell(cmd_mln(mln_path, dataspace, workspace, True))
    
    
def setup_mln_iteration(dataspace, iter_id=0, workspace=None):
    dataspace = ensure_dir(dataspace)
    workspace = ensure_dir(workspace / str(iter_id))
    shutil.copy(dataspace/"train_augmented.txt", workspace/"train_kge.txt")
    shutil.copy(dataspace/"hidden.txt", workspace/"hidden.txt")
    return workspace


def read_triple(datafile, entity2id=None, relation2id=None):
    '''
    Legacy Function. Read triples and map them into ids.
    '''
    identity = lambda x: x
    
    if entity2id is None: entity2id = identity
    if relation2id is None: entity2id = identity
    
    triples = pd.read_csv(datafile, sep="\t", header=None, index_col=None,
                          names=["subject", "predicate", "object"])
    triples = triples.transform({
        "subject": entity2id,
        "predicate": relation2id,
        "object": entity2id
    })
    return triples.to_numpy().tolist()


def read_txt_triples(datafile, entity_to_idx=True,
                     relation_to_idx=True):
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not existed: {datafile}")
    
    triples = pd.read_csv(datafile, sep="\t", header=None, index_col=None,
                          names=["subject", "predicate", "object"])
    
    # to_idx_dicts = None
    # to_entity_lists = None
    # converted_cols = dict()
    
    entities = pd.unique(pd.concat([triples["subject"], triples["object"]])).tolist()
    relations = pd.unique(triples["predicate"])
    
    if entity_to_idx:
        e2id_dict = {element : idx for idx, element in enumerate(entities)}
        def entity2id(entity):
            return e2id_dict[entity]
        triples["subject"] = triples["subject"].transform(entity2id)
        triples['object'] = triples['object'].transform(entity2id)
        
    if relation_to_idx:
        r2id_dict = {element : idx for idx, element in enumerate(relations)}
        def relation2id(relation):
            return r2id_dict[relation]
        triples["predicate"] = triples["predicate"].transform(relation2id)
            
    nentities = len(entities)
    nrelations = len(relations)
    
    return {
        "data": triples.to_numpy(),
        "nentities": nentities,
        "nrelations": nrelations,
        "entity2id": entity2id if entity_to_idx else None,
        "relation2id": relation2id if relation_to_idx else None,
    }
    

def build_dataset(workspace, negative_sample_size=4):
    workspace = ensure_dir(workspace).absolute()
    train_kge_file = workspace / "train_kge.txt"
    hidden_file = workspace / "hidden.txt"
    if not train_kge_file.exists() or not hidden_file.exists():
        raise FileNotFoundError("train_kge.txt and hidden.txt must be located at "
                                f"{workspace.absolute()}")
    
    train_triples_dict = read_txt_triples(train_kge_file)
    hidden_triples_dict = read_triple(hidden_file, 
                                      train_triples_dict["entity2id"], 
                                      train_triples_dict['relation2id'])
    
    _build_dataset = lambda triple_dict, mode: TrainDataset(
        triples = triple_dict["data"].tolist(),
        nentity = triple_dict["nentities"],
        nrelation = triple_dict["nrelations"],
        negative_sample_size = negative_sample_size,
        mode = mode
    )
    
    return _build_dataset(train_triples_dict, "head-batch"), \
        _build_dataset(train_triples_dict, "tail-batch")

    
def build_dataloader(dataset_tuple, batch_size=8, shuffle=True, num_workers=2):
    _build_dataloader = lambda dataset: DataLoader(
        dataset,
        batch_size,
        shuffle = shuffle,
        num_workers = max(1, num_workers//2),
        collate_fn = dataset.collate_fn
    )
    
    return BidirectionalOneShotIterator(
        _build_dataloader(dataset_tuple[0]),
        _build_dataloader(dataset_tuple[1])
    )
    

if __name__ == "__main__":
    data = "./data/neo4j/neo4j.txt"
    dataspace = ensure_dir("./record")
    workspace = ensure_dir(dataspace / "test_timestamp")
    workspace_1 = setup_mln_iteration(dataspace, 0, workspace)
    
    triples = read_txt_triples(workspace_1 / "train_kge.txt")
    print(triples["data"])
    print(triples["nentities"])
    print(triples["converted_cols"])
    
    datasets = build_dataset(workspace_1)
    print(datasets[0][0], data[1][0], sep='\n')
    dataloader = build_dataloader(datasets)
    print(next(dataloader))