import config as cfg
from utils import preprocess
from graph_data import GithubDataset
from train_nn import GCN, GraphSAGE, train_gnn
from logger import logger

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _naive_bayes(df, labels, relations, **args):
    clf = MultinomialNB()
    X_train, X_test, y_train, y_test = preprocess(df, labels)

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    logger.info(
        "{}: train_acc: {:.3f}, test acc: {:.3f}, train_f1: {:.3f}, test_f1: {:.3f}".format(
            args['model'], train_acc, test_acc, train_f1, test_f1
        )
    )

    return clf


def _logistic_regression(df, labels, relations, **args):
    clf = LogisticRegression(max_iter=args['lr_max_iter'])
    X_train, X_test, y_train, y_test = preprocess(df, labels)

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    logger.info(
        "{}: train_acc: {:.3f}, test acc: {:.3f}, train_f1: {:.3f}, test_f1: {:.3f}".format(
            args['model'], train_acc, test_acc, train_f1, test_f1
        )
    )

    return clf


def _gcn(df, labels, relations, **args):
    dataset = GithubDataset(undirected=args['undirected'])
    graph = dataset[0]
    graph = graph.add_self_loop()

    model = GCN(graph.ndata['feat'].shape[1], args['h_feats'], dataset.num_classes)
    train_gnn(graph, model, lr=args['lr'], epochs=args['epochs'])


def _graph_sage(df, labels, relations, **args):
    dataset = GithubDataset(undirected=args['undirected'])
    graph = dataset[0]
    graph = graph.add_self_loop()
    model = GraphSAGE(graph.ndata['feat'].shape[1], args['h_feats'], dataset.num_classes)
    train_gnn(graph, model, lr=args['lr'], epochs=args['epochs'])


models_dict = {
    'NaiveBayes': _naive_bayes,
    'LogisticRegression': _logistic_regression,
    'GCN': _gcn,
    'GraphSAGE': _graph_sage
}