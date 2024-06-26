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


def set_mln(mln_cpp, out_path=None):
    mln_cpp = Path(mln_cpp)
    if out_path is None:
        out_path = mln_cpp.parent / "mln"
    cmd = f"g++ -O3 {mln_cpp} -o {out_path} -lpthread -w"
    print(cmd)
    if os.system(cmd) == 0:
        return out_path
    else:
        raise RuntimeError(f"Compilation Failed.")


if __name__ == "__main__":
    cfg = load_yaml("./configs/default.yaml")
    print(cfg.mln_threshold_of_triplet)
    
    set_mln("./pLogicNet/mln/mln.cpp")