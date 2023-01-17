# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import itertools
import os
from os import listdir
from os.path import join
from pathlib import Path
import sys
import json
from typing import Any, List, Optional, Union
import numpy as np
import pickle
import torch
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn.functional as F


def load_text(text_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    texts, inds, sizes = [], [], []
    with open(text_path) as f:
        for ind, l in enumerate(f):
            l = l.strip()
            sz = len(l.split())
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                texts.append(l)
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    print(
        f"max_keep={max_keep}, min_keep={min_keep}, "
        f"loaded {len(texts)}, skipped {n_short} short and {n_long} long, "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
    )
    return texts, inds, tot, sizes


class ImageBatchData(Dataset):

    def __init__(self,
                 text_path,
                 ch2img_path,
                 feat_prefix,
                 pad_list: List[str] = None,
                 eos_list: List[str] = None,
                 label_processors: Optional[List[Any]] = None,
                 max_keep_sample_size: Optional[int] = None,
                 min_keep_sample_size: Optional[int] = None,
                 max_sample_size: Optional[int] = None,
                 shuffle: bool = True,
                 random_crop: bool = False,
                 single_target: bool = False,
    ):
        """
        Args:
            text_path: fairseq .trn file under the manifest directory
            ch2img_path: ch2img.json file storing a dict of [char]: list of images for the char
            feat_prefix: prefix of the feature .tsv .npy and .lengths files
            max_keep_sample_size: int, maximum sentence length
            min_keep_sample_size: int, minimum sentence length
            max_sample_size: int, maximum feature length
            shuffle: bool 
        """
        self.ch2img_path = ch2img_path
        with open(ch2img_path, 'r') as f:
            self.ch2img = json.load(f)

        self.texts, inds, tot, self.sizes = load_text(
            text_path, max_keep_sample_size, min_keep_sample_size
        )
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors

        assert label_processors is None

        self.feat_dict = {}
        feats = np.load(f"{feat_prefix}.npy")
        with open(f"{feat_prefix}.tsv", "r") as f_tsv:
            lines = f_tsv.read().strip().split("\n")
        _ = lines.pop(0)
        for i, line in enumerate(lines):
            img_id = Path(line.split("\t")[0]).stem
            self.feat_dict[img_id] = feats[i]
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        print(f"max_sample_size={self.max_sample_size}")
        self.sample_rate = 1

    def get_image_seq(self, index):
        text = self.texts[index]
        if not len(text):
            text = "_"
        feats = []
        for _ch in text.split():
            if _ch == "|":
                continue
            elif _ch == "_":
                _ch = "space"
            nch = len(self.ch2img[_ch])
            ix = random.randrange(nch)
            path = self.ch2img[_ch][ix]
            img_id = Path(path).stem
            feat = torch.tensor(
                self.feat_dict[img_id]
            )
            feats.append(feat)
        feats = torch.stack(feats)
        
        return feats

    def __getitem__(self, index):
        """
        Returns:
            images: seq_len x num_channels x height x width 
            lengths: length of the image sequence
        """
        images = self.get_image_seq(index)
        orig_size = len(images)
        images, start = self.crop_to_max_size(
            images, self.max_sample_size
        )
        # if len(images) != orig_size:
        #    print(f"Image exceeds max size: {orig_size} > {len(images)}")
        return images, index

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, image, target_size):
        size = len(image)
        diff = size - target_size
        if diff <= 0:
            return image, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return image[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        images = [s[0] for s in samples]
        
        if len(samples) == 0:
            return {}

        images = [s[0] for s in samples] # [T x 3 x 244 x 244] x B
        labels = [s[1] for s in samples]

        images = torch.nn.utils.rnn.pad_sequence(
            images, batch_first=True
        )
        labels = torch.tensor(labels)
        return images, labels

    def getDataLoader(self, batchSize, numWorkers=0, shuffle=False):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - numWorkers (int): number of workers
            - shuffle (bool): whether to shuffle the dataset or not
        """
        return data.DataLoader(
            self, 
            batch_size=batchSize,
            num_workers=numWorkers,
            shuffle=shuffle,
            collate_fn=self.collater,
        )
