import config as cfg
from utils import load_all_data
from models import models_dict
from logger import logger

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "NaiveBayes",
            "LogisticRegression",
            "GCN",
            "GraphSAGE"
        ],
        default="NaiveBayes",
    ),
    parser.add_argument(
        "--lr-max-iter",
        type=int,
        default=500,
    ),
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    ),
    parser.add_argument(
        "--h-feats",
        type=int,
        default=None
    ),
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    ),
    parser.add_argument(
        "--undirected",
        action='store_true'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    users, features, relations, labels = load_all_data()
    logger.info(args.__dict__)
    models_dict[args.model](features, labels, relations, **args.__dict__)