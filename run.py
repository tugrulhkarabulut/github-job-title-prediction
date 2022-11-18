import config as cfg
from utils import load_data
from models import models_dict

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "NaiveBayes",
            "LogisticRegression"
        ],
        default="NaiveBayes",
    ),
    return parser.parse_args()

def load_all_data():
    users = load_data(format='pandas')
    features = load_data(cfg.USER_FEATURES_FILE, format='pandas')
    relations = load_data(cfg.USER_RELATIONS_FILE, format='pandas')
    labels = load_data(cfg.USER_LABELS_FILE, format='pandas')

    users = users.set_index('login')
    features = features.set_index('username')
    labels = labels.set_index('login')



    return users, features, relations, labels


if __name__ == '__main__':
    args = parse_args()
    users, features, relations, labels = load_all_data()
    models_dict[args.model](features, labels)