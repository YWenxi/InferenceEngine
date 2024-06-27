import os
import yaml
import logging
from box import Box
from pathlib import Path


def load_yaml(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return Box(yaml_data)

def save_yaml(path, configs: Box):
    with open(path, "w") as f:
        yaml.dump(configs.to_dict(), f)
        

def ensure_dir(path):
    path = Path(path).absolute()
    path.mkdir(exist_ok=True, parents=True)
    return path


def shell(cmd):
    print(cmd)
    code = os.system(cmd)
    if code == 0:
        print("Succeed!")
    else: 
        raise RuntimeError(f"Failed. Exit Code: {code}")
    
    
def set_logger(log_file):
    '''
    Write logs to checkpoint and console
    '''

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    
def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


if __name__ == "__main__":
    cfg = load_yaml("./configs/default.yaml")
    print(cfg.mln_threshold_of_triplet)
    save_yaml("./cfg", cfg)