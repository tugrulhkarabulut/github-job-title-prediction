from logger import logger


import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv, SAGEConv

from sklearn.metrics import f1_score


class GCN(nn.Module):
    def __init__(self, in_feats, h_feat, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feat)
        self.conv2 = GraphConv(h_feat, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feat, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feat, "mean")
        self.conv2 = SAGEConv(h_feat, num_classes, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def train_gnn(g, model, lr=0.001, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0
    best_test_f1 = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    label_mask = g.ndata["label_mask"]
    train_label_mask = g.ndata["train_label_mask"]
    test_label_mask = g.ndata["test_label_mask"]
    for e in range(epochs):
        logits = model(g, features.float())

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_label_mask], labels[train_label_mask])

        train_acc = (pred[train_label_mask] == labels[train_label_mask]).float().mean()
        test_acc = (pred[test_label_mask] == labels[test_label_mask]).float().mean()
        train_f1 = f1_score(
            labels[train_label_mask], pred[train_label_mask], average="weighted"
        )
        test_f1 = f1_score(
            labels[test_label_mask], pred[test_label_mask], average="weighted"
        )

        if best_test_acc < test_acc:
            best_test_acc = test_acc

        if best_test_f1 < test_f1:
            best_test_f1 = test_f1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 50 == 0:
            logger.info(
                "In epoch {}, loss: {:.3f}, train_acc: {:.3f}, test acc: {:.3f} (best {:.3f}), train_f1: {:.3f}, test_f1: {:.3f} (best: {:.3f})".format(
                    e + 1,
                    loss,
                    train_acc,
                    test_acc,
                    best_test_acc,
                    train_f1,
                    test_f1,
                    best_test_f1,
                )
            )
