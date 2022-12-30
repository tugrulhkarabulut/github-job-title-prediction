import config as cfg
from utils import load_all_data, update_experiments
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
        help="Model name."
    ),
    parser.add_argument(
        "--lr-max-iter",
        type=int,
        default=500,
        help="Logistic Regression iteration."
    ),
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for GCNs."
    ),
    parser.add_argument(
        "--h-feats",
        type=int,
        default=None,
        help="Hidden units."
    ),
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs."
    ),
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Number of iterations to wait for improvement before early stopping."
    )
    parser.add_argument(
        "--undirected",
        action='store_true',
        help="Make the graph undirected."
    )
    parser.add_argument(
        "--feature-selection",
        choices=[None, "variance", "select_from_model"],
        help="Feature selection method."
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-3,
        help="Threshold value for variance feature selection."
    )
    parser.add_argument(
        "--select-from",
        type=str,
        default="svc",
        choices=["svc", "extra_trees"],
        help="Select features according to given model."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        choices=[1,5],
        help="Number of splits for k-fold cross-validation."
    )
    parser.add_argument(
        "--neighborhood-features",
        type=str,
        default='mean',
        choices=['mean', 'max'],
        help="Neighborhood aggregation function for non-graph models."
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    users, features, relations, labels = load_all_data()
    logger.info(args.__dict__)
    mean_test_score, std_test_score = run_pipeline(features, labels, relations, **args.__dict__)
    exp_data = args.__dict__.copy()
    exp_data['mean_test_score'] = mean_test_score
    exp_data['std_test_score'] = std_test_score
    update_experiments(exp_data)