#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import os.path as osp
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
np.random.seed(1)
torch.manual_seed(1)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Linear separability score for features"
    )
    # fmt: off
    parser.add_argument('feat_dir', help='location of features and metadata')

    # fmt: on
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = torch.FloatTensor(
        np.load(osp.join(args.feat_dir, "train.npy"))
    ).to(device)
    X_test = torch.FloatTensor(
        np.load(osp.join(args.feat_dir, "valid.npy"))
    ).to(device)
    train_labels = []
    test_labels = []
    vocab = []
    with open(osp.join(args.feat_dir, "train.wrd"), "r") as f_tr,\
        open(osp.join(args.feat_dir, "valid.wrd"), "r") as f_tx:
        for sent in f_tr:
            for w in sent.rstrip("\n").split():
                if not w in vocab:
                    train_labels.append(len(vocab))
                    vocab.append(w)
                else:
                    train_labels.append(vocab.index(w))
        
        for sent in f_tx:
            for w in sent.rstrip("\n").split():
                if not w in vocab:
                    print(f"Warning: {w} not in training set", flush=True)
                    test_labels.append(len(vocab))
                    vocab.append(w)
                else:
                    test_labels.append(vocab.index(w))
    y_train = torch.LongTensor(train_labels).to(device)
    y_test = torch.LongTensor(test_labels).to(device)
    assert (X_train.size(0) == y_train.size(0)) and (X_test.size(0) == y_test.size(0))
    clf = nn.Linear(X_train.shape[-1], len(vocab))
    clf = clf.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        clf.parameters(),
        lr=0.001,
    )
    for epoch in range(100):
        logits = clf(X_train)
        loss = criterion(logits, y_train)
        print(f"Epoch {epoch}, training loss: {loss:.3f}", flush=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred = clf(X_test).argmax(-1).detach().cpu().numpy()
        acc = accuracy_score(y_test.cpu(), y_pred)
        micro_f1 = f1_score(y_test.cpu(), y_pred, average="micro")
        macro_f1 = f1_score(y_test.cpu(), y_pred, average="macro")
        print(
            f"Validation Epoch {epoch}, accuracy: {acc:.4f}\tmicro F1: {micro_f1:.4f}\tmacro F1: {macro_f1:.4f}", 
            flush=True,
        )

if __name__ == "__main__":
    main()
