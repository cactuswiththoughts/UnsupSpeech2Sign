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
import pdb
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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


class VideoBatchData(Dataset):

    def __init__(self,
                 text_path,
                 wrd2vid_path,
                 feat_prefix,
                 pad_list: List[str] = None,
                 eos_list: List[str] = None,
                 label_processors: Optional[List[Any]] = None,
                 max_keep_sample_size: Optional[int] = None,
                 min_keep_sample_size: Optional[int] = None,
                 max_sample_size: Optional[int] = None,
                 shuffle: bool = True,
                 random_crop: bool = False,
                 return_wrd_ids: bool = False,
    ):
        """
        Args:
            text_path: fairseq .trn file under the manifest directory
            wrd2vid_path: wrd2vid.json file storing a dict of [word]: list of videos for the word
            feat_prefix: prefix of the feature .tsv .npy and .lengths files
            max_keep_sample_size: int, maximum sentence length
            min_keep_sample_size: int, minimum sentence length
            max_sample_size: int, maximum feature length
            shuffle: bool 
        """
        self.wrd2vid_path = wrd2vid_path
        with open(wrd2vid_path, 'r') as f:
            self.wrd2vid = json.load(f)

        self.texts, inds, tot, self.sizes = load_text(
            text_path, max_keep_sample_size, min_keep_sample_size
        )
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.return_wrd_ids = return_wrd_ids

        self.feat_dict = {}
        feats = np.load(f"{feat_prefix}.npy")
        self.d = feats.shape[-1]
        with open(f"{feat_prefix}.tsv", "r") as f_tsv,\
            open(f"{feat_prefix}.lengths", "r") as f_len:
            lines = f_tsv.read().strip().split("\n")
            sizes = f_len.read().strip().split("\n")
            sizes = map(int, sizes)            
        _ = lines.pop(0)
        offset = 0
        for line, size in zip(lines, sizes):
            wrd_id = Path(line.split("\t")[0]).name
            self.feat_dict[wrd_id] = feats[offset:offset+size]
            offset += size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        print(f"max_sample_size={self.max_sample_size}")
        self.sample_rate = 1

    def get_video_seq(self, index):
        text = self.texts[index]
        if not len(text):
            text = "_"
        feats = []
        sizes = []
        wrd_ids = []
        for w in text.split():
            nw = len(self.wrd2vid[w])
            ix = random.randrange(nw)
            wrd_id = None
            while (not wrd_id in self.feat_dict) and (ix < nw):
                wrd_info = self.wrd2vid[w][ix]
                vid_id = wrd_info["url"].split("v=")[-1]
                wrd_id = f"{vid_id}.mp4_{wrd_info['start']}_{wrd_info['end']}_{wrd_info['text']}"
                if not wrd_id in self.feat_dict:
                    wrd_id = f"{vid_id}.mp4_{wrd_info['start_time']}_{wrd_info['end_time']}"
                ix += 1
                
            if not wrd_id in self.feat_dict:
                print(f"Failed to find video feature for sign {wrd_id}, {sorted(self.feat_dict)[:3]}")
                feats.append(torch.zeros(self.d))
                continue
            feat = torch.tensor(
                self.feat_dict[wrd_id]
            )
            feats.append(feat)
            sizes.append(len(feat))
            wrd_ids.append(wrd_id)
        feats = torch.cat(feats)
        sizes = torch.tensor(sizes)
        return feats, sizes, wrd_ids

    def __getitem__(self, index):
        """
        Returns:
            images: seq_len x num_channels x height x width 
            lengths: length of the image sequence
        """
        images, sizes, wrd_ids = self.get_video_seq(index)
        orig_size = len(images)
        images, start = self.crop_to_max_size(
            images, self.max_sample_size
        )
        return images, sizes, wrd_ids

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
        images = [s[0] for s in samples]
        
        if len(samples) == 0:
            return {}

        images = [s[0] for s in samples] # [T x D] x B
        labels = [s[1] for s in samples]
        wrd_ids = [s[2] for s in samples]

        images = torch.nn.utils.rnn.pad_sequence(
            images, batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True
        )
        labels = torch.tensor(labels)
        
        if self.return_wrd_ids:
            return images, labels, wrd_ids
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
