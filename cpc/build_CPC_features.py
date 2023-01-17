import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time
import numpy as np

import torch
from cpc.image_dataset import ImageBatchData
from cpc.feature_loader import buildFeature, FeatureModule, loadModel

from utils.utils_functions import writeArgs, loadCPCFeatureMaker

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export CPC features from audio files.')
    parser.add_argument('pathCPCCheckpoint', type=str,
                        help='Path to the CPC checkpoint.')
    parser.add_argument('pathDB', type=str,
                        help='Path to the dataset that we want to quantize.')
    parser.add_argument('ch2img_path', type=str, 
                        help='Path to the mapping from characters to images.')
    parser.add_argument('pathFeat', type=str, default=None,
                        help='Path to the directory containing the '
                        'features.')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('--max_keep_sample_size', type=int, default=None,
                          help='Maximal sentence length'
                          'in characters.')
    parser.add_argument('--min_keep_sample_size', type=int, default=None,
                          help='Minimal sentence length'
                          'in characters.')
    parser.add_argument('--max_sample_size', type=int, default=None,
                          help='Maximal feature length'
                          'in characters.')
    parser.add_argument('--split', type=str, default='test', 
                        help='Split of the dataset')
    parser.add_argument('--file_extension', type=str, default="wav",
                          help="Extension of the audio files in the dataset (default: wav).")
    parser.add_argument('--get_encoded', type=bool, default=False,
                        help='If True, get the outputs of the encoder layer only (default: False).')
    parser.add_argument('--gru_level', type=int, default=-1,
                        help='Hidden level of the LSTM autoregressive model to be taken'
                        '(default: -1, last layer).')
    parser.add_argument('--max_size_seq', type=int, default=64000,
                        help='Maximal number of frames to consider in each chunk'
                        'when computing CPC features (defaut: 64000).')
    parser.add_argument('--seq_norm', type=bool, default=False,
                        help='If True, normalize the output along the time'
                        'dimension to get chunks of mean zero and var 1 (default: False).')
    parser.add_argument('--strict', type=bool, default=True,
                        help='If True, each batch of feature '
                        'will contain exactly max_size_seq frames (defaut: True).')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)
    print("=============================================================", flush=True)
    print(f"Building CPC features from {args.pathDB}", flush=True)
    print("=============================================================", flush=True)

    # Find all sequences
    data_path = Path(args.pathDB)
    feat_path = Path(args.pathFeat)
    ch2img_path = Path(args.ch2img_path)
    testDataset = ImageBatchData(
        data_path / f"{args.split}.trn",
        ch2img_path,
        feat_path,
    )
    testLoader = testDataset.getDataLoader(batchSize=1)
    
    # Verify the output directory
    print("", flush=True)
    print(f"Creating the output directory at {args.pathOutputDir}", flush=True)
    Path(args.pathOutputDir).mkdir(parents=True, exist_ok=True)
    
    # Load CPC feature maker
    print("", flush=True)
    print(f"Loading CPC featureMaker from {args.pathCPCCheckpoint}", flush=True)
    featureMaker = loadCPCFeatureMaker(
        args.pathCPCCheckpoint,
        gru_level = args.gru_level,
        get_encoded = args.get_encoded,
        keep_hidden = True,
        device = "cpu" if args.debug or args.cpu else "cuda")
    featureMaker.eval()
    if not args.cpu:
        featureMaker.cuda()
    print("CPC FeatureMaker loaded!", flush=True)

    # Building features
    print("", flush=True)
    print(f"Building CPC features and saving outputs to {args.pathOutputDir}...", flush=True)
    bar = progressbar.ProgressBar(maxval=len(testDataset))
    bar.start()
    start_time = time()
    CPC_features = []
    lengths = [] 
    with torch.no_grad():
        for index, vals in enumerate(testLoader):
            if args.debug and index > 10:
                break
            bar.update(index)

            # Computing features
            feat = featureMaker(vals)[0].cpu().numpy()
            CPC_features.append(feat)
            lengths.append(str(len(feat)))
    # Save the outputs
    CPC_features = np.concatenate(CPC_features)

    file_npy = os.path.join(args.pathOutputDir, f"{args.split}.npy")
    np.save(file_npy, CPC_features)
    file_len = os.path.join(args.pathOutputDir, f"{args.split}.lengths")
    with open(file_len, "w") as f_len:
        f_len.write("\n".join(lengths))

    bar.finish()
    print(f"...done {len(testDataset)} files in {time()-start_time} seconds.", flush=True)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
