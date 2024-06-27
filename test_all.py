from utils import ensure_dir

from models import pLogicNet

from data import build_dataloader
from data import build_dataset
from data import read_txt_triples
from data import data_preprocessing
from data import setup_mln_iteration

from utils import load_yaml

def test_dataset():
    data = "./data/neo4j/neo4j.txt"
    dataspace = ensure_dir("./record")
    workspace = ensure_dir(dataspace / "test_timestamp")
    data_preprocessing("./pLogicNet/mln/mln", data, dataspace)
    setup_mln_iteration(dataspace, workspace=workspace)
    workspace_1 = setup_mln_iteration(dataspace, 1, workspace)
    
    triples = read_txt_triples(workspace_1 / "train_kge.txt")
    # print(triples["data"])
    # print(triples["nentities"])
    # print(triples["converted_cols"])
    
    datasets = build_dataset(workspace_1)
    # print(datasets[0][0], data[1][0], sep='\n')
    dataloader = build_dataloader(datasets)
    # print(next(dataloader))

def test_model():
    args = load_yaml("./configs/default.yaml")
    dataspace = ensure_dir("./record")
    workspace = ensure_dir(dataspace / "test_timestamp" / "0")
    
    triples = read_txt_triples(workspace / "train_kge.txt")
    model = pLogicNet(triples["nentities"], triples["nrelations"])
    datasets = build_dataset(workspace)
    dataloader = build_dataloader(datasets)
    model.train(dataloader, args=args, max_steps=10)