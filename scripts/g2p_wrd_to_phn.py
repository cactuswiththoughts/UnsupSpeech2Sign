#!/usr/bin/env python3 -u

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrd_path')
    parser.add_argument('--in_path')
    parser.add_argument('--out_path')
    parser.add_argument('--comma_separated', action="store_true")
    args = parser.parse_args()

    with open(args.wrd_path, 'r') as f_wrd,\
         open(args.in_path, 'r') as f_in,\
         open(args.out_path, 'w') as f_out:
        
        wrd_to_phn = dict()
        for line in f_wrd:
            for w in line.split():
                w = w.upper()
                if not w in wrd_to_phn:
                    wrd_to_phn[w] = []
                    if args.comma_separated:
                        phns = w.split(",")
                    else:
                        phns = list(w)
                    for phn in phns:
                        wrd_to_phn[w].append(phn)
        
        for line in f_in:
            w = line.strip()
            f_out.write(' '.join(wrd_to_phn[w])+'\n')
    
if __name__ == '__main__':
    main()
