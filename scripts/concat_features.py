import argparse
import numpy as np
from pathlib import Path
from shutil import copyfile

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir1")
    parser.add_argument("--in_dir2")
    parser.add_argument("--out_dir")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    in_dir1 = Path(args.in_dir1)
    in_dir2 = Path(args.in_dir2)
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    for fn in in_dir1.iterdir():
        if fn.name.endswith(".lengths") or fn.name.endswith(".tsv"):
            copyfile(fn, out_dir / fn.name)
        
        if fn.name.endswith(".npy"):
            f1 = np.load(fn)
            f2 = np.load(in_dir2 / fn.name)
            f = np.concatenate((f1, f2), axis=-1)
            print(f.shape)
            np.save(out_dir / fn.name, f)

if __name__ == "__main__":
    main()
