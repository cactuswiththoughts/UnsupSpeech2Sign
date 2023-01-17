#!/usr/bin/env python3 -u
import argparse
import numpy as np
from pathlib import Path
import torch

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="Find the maximum segment length of a dataset"
    )
    # fmt: off
    parser.add_argument('source', help='where the features are')
    parser.add_argument('--cluster-dir', help='where the clusters are')
    parser.add_argument('--fmt', default='sklearn', choices={"sklearn", "faiss"})
    # fmt: on

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    source = Path(args.source)
    cluster_dir = Path(args.cluster_dir)
    l_max = 1
    for split in ["train", "valid"]:
        offsets = []
        sizes = []
        with open(source / f"{split}.lengths", "r") as len_f:
            offset = 0
            for line in len_f:
                length = int(line.rstrip())
                sizes.append(length)
                offsets.append(offset)
                offset += length
        
        with open(cluster_dir / f"{split}.src", "r") as cf:
            if args.fmt == "sklearn":
                lines = cf.read().strip().split("\n")
                for size, offset in zip(sizes, offsets):
                    items = lines[offset:offset+size]
                    clust = list(map(int, items))  
                    clust = torch.LongTensor(clust)
                    _, counts = clust.unique_consecutive(return_counts=True)
                    l = counts.max().item()
                    if l > l_max:
                        l_max = l
            else:
                for line, size in zip(cf, sizes):
                    line = line.rstrip()
                    items = line.split()
                    if len(items) < size:
                        items.append(items[-1])
                    else:
                        items = items[:size]
                    clust = list(map(int, items))
                    clust = torch.LongTensor(clust)
                    _, counts = clust.unique_consecutive(return_counts=True)
                    l = counts.max().item()
                    if l > l_max:
                        l_max = l
    print(f"Max segment length: {l_max}")
    with open(cluster_dir / "max_segment_length.txt", "w") as out_f:
        print(l_max, file=out_f)

if __name__ == "__main__":
    main()