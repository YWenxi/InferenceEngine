import os
from torch.utils.data import Dataset
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
    

if __name__ == "__main__":
    data = "./data/neo4j/neo4j.txt"
    dataspace = ensure_dir("./record")
    workspace = ensure_dir(dataspace / "test_timestamp")
    data_preprocessing("./pLogicNet/mln/mln", data, dataspace)
    setup_mln_iteration(dataspace, workspace=workspace)
    setup_mln_iteration(dataspace, 1, workspace)