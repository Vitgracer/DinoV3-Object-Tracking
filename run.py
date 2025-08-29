import yaml
from argparse import Namespace

def run_tracking(config):
    pass

if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = Namespace(**yaml.safe_load(file))
    run_tracking(config)