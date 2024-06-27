import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from pLogicNet.utils import cmd_mln
from pLogicNet.kge.dataloader import TrainDataset, BidirectionalOneShotIterator

import shutil
import datetime
from utils import ensure_dir, shell


def data_preprocessing(mln_path, data, dataspace="./record"):
    dataspace = ensure_dir(dataspace)
    
    shutil.copy(data, dataspace / "train.txt")
    shutil.copy(data, dataspace / "train_augmented.txt")
    
    shell(cmd_mln(mln_path, dataspace, dataspace, True))
    
    
def get_timestamp():
    return str(datetime.datetime.now()).replace(' ', '_')
    
    
def setup_mln_iteration(dataspace, iter_id=0, workspace=None):
    dataspace = ensure_dir(dataspace)
    workspace = ensure_dir(workspace / str(iter_id))
    shutil.copy(dataspace/"train_augmented.txt", workspace/"train_kge.txt")
    shutil.copy(dataspace/"hidden.txt", workspace/"hidden.txt")
    return workspace


def read_txt_triples(datafile):
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not existed: {datafile}")
    
    triples = pd.read_csv(datafile, sep=" ", header=None, index_col=None,
                          names=["subject", "predicate", "object"])
    
    to_idx_dicts = dict()
    to_entity_lists = dict()
    converted_cols = []
    
    for col in triples:
        if triples[col].dtype != "int64":
            uniques = pd.unique(triples[col]).tolist()
            uniques_to_id_dict = {element : idx for idx, element in enumerate(uniques)}
            triples[col] = triples[col].apply(
                lambda element: uniques_to_id_dict[element]
            )
            to_idx_dicts[col] = uniques_to_id_dict
            to_entity_lists[col] = uniques
            converted_cols.append(col)
            
    nentities = len(pd.unique(triples['subject'])) + len(pd.unique(triples['object']))
    nrelations = len(pd.unique(triples['predicate']))
    
    return {
        "data": triples.to_numpy(),
        "nentities": nentities,
        "nrelations": nrelations,
        "to_idx_dicts": to_idx_dicts,
        "to_entity_lists": to_entity_lists,
        "converted_cols": converted_cols
    }
    

def build_dataset(workspace, negative_sample_size=4):
    workspace = ensure_dir(workspace).absolute()
    train_kge_file = workspace / "train_kge.txt"
    hidden_file = workspace / "hidden.txt"
    if not train_kge_file.exists() or not hidden_file.exists():
        raise FileNotFoundError("train_kge.txt and hidden.txt must be located at "
                                f"{workspace.absolute()}")
    
    train_triples_dict = read_txt_triples(train_kge_file)
    hidden_triples_dict = read_txt_triples(hidden_file)
    
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
    data_preprocessing("./pLogicNet/mln/mln", data, dataspace)
    setup_mln_iteration(dataspace, workspace=workspace)
    workspace_1 = setup_mln_iteration(dataspace, 1, workspace)
    
    triples = read_txt_triples(workspace_1 / "train_kge.txt")
    print(triples["data"])
    print(triples["nentities"])
    print(triples["converted_cols"])
    
    datasets = build_dataset(workspace_1)
    print(datasets[0][0], data[1][0], sep='\n')
    dataloader = build_dataloader(datasets)
    print(next(dataloader))