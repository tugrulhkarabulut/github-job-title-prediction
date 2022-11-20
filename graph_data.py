from utils import load_all_data, preprocess

import torch

import dgl
from dgl.data import DGLDataset


import pandas as pd
from sklearn.preprocessing import LabelEncoder


class GithubDataset(DGLDataset):
    def __init__(self, undirected=False):
        self.undirected = undirected
        super().__init__(name='github_dataset')

    def process(self):
        _, features, relations, labels = load_all_data()
        X_train, X_test, y_train, y_test = preprocess(features, labels, include_unlabeled=True, test_size=0.2)
        features = pd.concat([X_train, X_test], axis=0)
        features = features.sort_index()
        labels = pd.concat([y_train, y_test], axis=0)
        labels = labels.sort_index()

        all_users = list(set(X_train.index).union(X_test.index))
        l_user = LabelEncoder()
        l_user.fit(all_users)

        l_label = LabelEncoder()
        labels = l_label.fit_transform(labels)


        src = l_user.transform(relations['following'])
        dest = l_user.transform(relations['follow'])

        train_index = l_user.transform(X_train.index)
        test_index = l_user.transform(X_test.index)


        edges_src = torch.from_numpy(src)
        edges_dst = torch.from_numpy(dest)
        
        self.l_user = l_user
        self.l_label = l_label
        self.num_classes = len(l_label.classes_) - 1

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=features.shape[0])
        if self.undirected:
            self.graph.add_edges(edges_dst, edges_src)
        
        node_features = torch.from_numpy(features.to_numpy())
        node_labels = torch.from_numpy(labels)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        n_nodes = features.shape[0]
        n_train = int(n_nodes * 0.8)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_index] = True
        test_mask[test_index] = True

        labeled_mask = (node_labels < self.num_classes)
        train_labeled_mask = (labeled_mask) & (train_mask)
        test_labeled_mask = (labeled_mask) & (test_mask)


        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata['label_mask'] = labeled_mask
        self.graph.ndata['train_label_mask'] = train_labeled_mask
        self.graph.ndata['test_label_mask'] = test_labeled_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1