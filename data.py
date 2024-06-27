import os

from torch.utils.data import Dataset
import pandas as pd

from pLogicNet.utils import cmd_mln
from pLogicNet.kge.dataloader import TrainDataset

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
    to_element_lists = dict()
    converted_cols = []
    
    for col in triples:
        if triples[col].dtype != "int64":
            uniques = pd.unique(triples[col]).tolist()
            uniques_to_id_dict = {element : idx for idx, element in enumerate(uniques)}
            triples[col] = triples[col].apply(
                lambda element: uniques_to_id_dict[element]
            )
            to_idx_dicts[col] = uniques_to_id_dict
            to_element_lists[col] = uniques
            converted_cols.append(col)
    
    return {
        "data": triples.to_numpy(), 
        "to_idx_dicts": to_idx_dicts,
        "to_element_lists": to_element_lists,
        "converted_cols": converted_cols
    }
    

if __name__ == "__main__":
    data = "./data/neo4j/neo4j.txt"
    dataspace = ensure_dir("./record")
    workspace = ensure_dir(dataspace / "test_timestamp")
    data_preprocessing("./pLogicNet/mln/mln", data, dataspace)
    setup_mln_iteration(dataspace, workspace=workspace)
    workspace_1 = setup_mln_iteration(dataspace, 1, workspace)
    
    triples = read_txt_triples(workspace_1 / "train_kge.txt")
    print(triples["data"])
    print(triples["converted_cols"])