import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    vocab = []
    with open(args.in_path, "r") as fin,\
        open(args.out_path, "w") as fout:
        for line in fin:
            phns = line.rstrip().split()
            phn_ints = []
            for phn in phns:
                if phn not in vocab:
                    vocab.append(phn)
                phn_ints.append(str(vocab.index(phn)))
            fout.write(" ".join(phn_ints)+"\n")

if __name__ == "__main__":
    main()
