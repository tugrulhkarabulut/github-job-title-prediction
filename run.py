import config as cfg
from utils import load_all_data
from models import run_pipeline
from logger import logger

import argparse


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
        "--patience",
        type=int,
        default=50
    )
    parser.add_argument(
        "--undirected",
        action='store_true'
    )
    parser.add_argument(
        "--feature-selection",
        choices=[None, "variance", "select_from_model"]
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--select-from",
        type=str,
        default="svc",
        choices=["svc", "extra_trees"]
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        choices=[1,5]
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    users, features, relations, labels = load_all_data()
    logger.info(args.__dict__)
    run_pipeline(features, labels, relations, **args.__dict__)