import os
import yaml
from box import Box
from pathlib import Path


def load_yaml(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return Box(yaml_data)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(exist_ok=True)
    return path


def shell(cmd):
    print(cmd)
    code = os.system(cmd)
    if code == 0:
        print("Succeed!")
    else: 
        raise RuntimeError(f"Failed. Exit Code: {code}")


if __name__ == "__main__":
    cfg = load_yaml("./configs/default.yaml")
    print(cfg.mln_threshold_of_triplet)