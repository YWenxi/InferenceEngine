import yaml
from box import Box


def load_yaml(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return Box(yaml_data)


if __name__ == "__main__":
    cfg = load_yaml("./configs/default.yaml")
    print(cfg.mln_threshold_of_triplet)