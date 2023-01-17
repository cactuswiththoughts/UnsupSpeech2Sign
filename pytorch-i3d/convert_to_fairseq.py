import argparse
import json
import numpy as np
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--pooling")
    return parser
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    features = []
    with open(out_dir / "train.tsv", "w") as f_tsv,\
        open(out_dir / "train.lengths", "w") as f_len,\
        open(out_dir / "train.src", "w") as f_src:
        print(args.out_dir, file=f_tsv)
        for fn in in_dir.iterdir():
            if fn.name.endswith(".npy"):
                if ".mp4" in fn.name:
                    video_id, desc = fn.name.split(".mp4")
                else:
                    video_id, desc = fn.name.split(".mkv")
                desc = desc.rstrip(".npy")
                new_fn = f"{video_id}.mp4{desc}"
                print("\t".join([new_fn, str(1)]), file=f_tsv)
                
                feat = np.load(fn)
                if args.pooling == "mean":
                    feat = feat.mean(0, keepdims=True)
                print(len(feat), file=f_len)
                print(" ".join(["1"]*len(feat)), file=f_src)
                features.append(feat)
    
    features = np.concatenate(features)
    np.save(out_dir / "train.npy", features)

if __name__ == "__main__":
    main()
    
    
