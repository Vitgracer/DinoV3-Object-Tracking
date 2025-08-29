import yaml
from argparse import Namespace
from interaction.coordinate import get_tracking_coordinate


def run_tracking(config):
    tracking_coordinate, tacking_features = get_tracking_coordinate(config)


if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = Namespace(**yaml.safe_load(file))
    run_tracking(config)