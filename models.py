import config as cfg
from utils import preprocess
from graph_data import GithubDataset
from train_nn import GCN, GraphSAGE, train_gnn
from logger import logger

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel


def build_graph(X_train, X_test, y_train, y_test, X_unlabeled, relations, args):
    dataset = GithubDataset(
        X_train,
        X_test,
        y_train,
        y_test,
        X_unlabeled,
        relations,
        undirected=args["undirected"],
    )
    graph = dataset[0]
    logger.info(f"#nodes: {graph.num_nodes()}, #edges: {graph.num_edges()}")
    graph = graph.add_self_loop()

    return dataset, graph


def evaluate(y_train, y_pred_train, y_test, y_pred_test):
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted")
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")

    logger.info(
        "train_acc: {:.3f}, test acc: {:.3f}, train_f1: {:.3f}, test_f1: {:.3f}".format(
            train_acc, test_acc, train_f1, test_f1
        )
    )

    return train_acc, test_acc, train_f1, test_f1


def feature_selection(X_train, y_train, args):
    if args["feature_selection"] == "variance":
        vt = VarianceThreshold(threshold=args["variance_threshold"])
        X_train = pd.DataFrame(
            vt.fit_transform(X_train),
            index=X_train.index,
            columns=vt.get_feature_names_out(),
        )
        return X_train, vt

    if args["feature_selection"] == "select_from_model":
        if args["select_from"] == "svc":
            select_from = LinearSVC(penalty="l1", dual=False, random_state=42)
        elif args["select_from"] == "extra_trees":
            select_from = ExtraTreesClassifier(random_state=42)

        sfm = SelectFromModel(select_from)
        X_train = pd.DataFrame(
            sfm.fit_transform(X_train, y_train),
            index=X_train.index,
            columns=sfm.get_feature_names_out(),
        )
        return X_train, sfm

    return X_train, None


def _naive_bayes(X_train, y_train, **args):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    return clf


def _logistic_regression(X_train, y_train, **args):
    clf = LogisticRegression(max_iter=args["lr_max_iter"])
    clf.fit(X_train, y_train)

    return clf


def _gcn(graph, dataset, **args):
    model = GCN(graph.ndata["feat"].shape[1], args["h_feats"], dataset.num_classes)
    train_gnn(graph, model, lr=args["lr"], epochs=args["epochs"])
    return model


def _graph_sage(graph, dataset, **args):
    model = GraphSAGE(
        graph.ndata["feat"].shape[1], args["h_feats"], dataset.num_classes
    )
    train_gnn(graph, model, lr=args["lr"], epochs=args["epochs"])
    return model


def train_model(X_train, y_train, args):
    model = models_dict[args["model"]](X_train, y_train, **args)
    return model


def train_graph_model(graph, dataset, args):
    model = models_dict[args["model"]](graph, dataset, **args)
    return model


def run_pipeline(df, labels, relations, **args):
    logger.info(f"examples: {df.shape[0]}, features: {df.shape[1]}")

    is_graph_model = True if args["model"] in ["GCN", "GraphSAGE"] else False
    X_trains, X_tests, y_trains, y_tests, X_unlabeled = preprocess(
        df,
        labels,
        include_unlabeled=is_graph_model,
        n_splits=args['n_splits']
    )

    for i, (X_train, X_test, y_train, y_test) in enumerate(
        zip(X_trains, X_tests, y_trains, y_tests)
    ):
        logger.info(
            f"split {i+1}: train: {X_train.shape[0]}, test: {X_test.shape[0]}"
        )

        X_train, feature_selector = feature_selection(X_train, y_train, args)
        if feature_selector is not None:
            X_test = pd.DataFrame(
                feature_selector.transform(X_test),
                index=X_test.index,
                columns=feature_selector.get_feature_names_out(),
            )
            X_unlabeled = pd.DataFrame(
                feature_selector.transform(X_unlabeled),
                index=X_unlabeled.index,
                columns=feature_selector.get_feature_names_out(),
            )
            logger.info(f"After feature selection: {X_train.shape[1]} features.")

        if is_graph_model:
            dataset, graph = build_graph(
                X_train, X_test, y_train, y_test, X_unlabeled, relations, args
            )
            model = train_graph_model(graph, dataset, args)
            model.eval()
            y_pred = model(graph, graph.ndata["feat"].float())
            y_pred_train = y_pred[graph.ndata["train_label_mask"]].argmax(1).numpy()
            y_pred_test = y_pred[graph.ndata["test_label_mask"]].argmax(1).numpy()
            y_pred_train = dataset.l_label.inverse_transform(y_pred_train)
            y_pred_test = dataset.l_label.inverse_transform(y_pred_test)
            y_train = y_train.sort_index()
            y_test = y_test.sort_index()
        else:
            model = train_model(X_train, y_train, args)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        evaluate(y_train, y_pred_train, y_test, y_pred_test)


models_dict = {
    "NaiveBayes": _naive_bayes,
    "LogisticRegression": _logistic_regression,
    "GCN": _gcn,
    "GraphSAGE": _graph_sage,
}
