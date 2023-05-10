import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation)
        )
        for _ in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        self.layers.append(
            GraphConv(n_hidden, n_classes)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(n_epochs=100, lr=1e-2, weight_decay=5e-4, n_hidden=16, n_layers=1,
          activation=F.relu, dropout=0.5):
    data = CoraGraphDataset()
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    model = GCN(
        g=g,
        in_feats=in_feats,
        n_hidden=n_hidden,
        n_classes=n_classes,
        n_layers=n_layers,
        activation=activation,
        dropout=dropout
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    for epoch in range(n_epochs):
        model.train()
        logits = model(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {} | loss {:.5f} | Accuracy{:.4f} "
              .format(epoch, loss, acc))
    print()
    acc = evaluate(model, features, labels, test_mask)
    print('Test accuracy {:.2%}'.format(acc))


print('1111111')
train()
