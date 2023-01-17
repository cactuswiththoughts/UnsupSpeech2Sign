import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_path",
        help=".wrd input file",
    )
    parser.add_argument(
        "out_path",
        help=".phn output file with character sequence",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.in_path, "r") as fin,\
        open(args.out_path, "w") as fout:
        for line in fin:
            sent = [
                c.upper() for w in line.strip().split() for c in w
            ]
            fout.write(" ".join(sent)+"\n")

if __name__ == "__main__":
    main()

