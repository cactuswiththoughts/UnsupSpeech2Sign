import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", "-i")
    parser.add_argument("--mapping", "-m")
    parser.add_argument("--to", type=int, default=39, choices={39, 48})
    parser.add_argument("--out_path", "-o")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    mapping = {}
    with open(args.mapping, "r") as fm:
        for l in fm:
            if len(l.strip().split("\t")) > 1:
                phn60, phn48, phn39 = l.strip().split("\t")
            else:
                phn60 = l.strip()
                phn48 = 'sil'
                phn39 = 'sil'

            if args.to == 39:
                mapping[phn60] = phn39
            else:
                mapping[phn60] = phn48

    with open(args.in_path, "r") as fin,\
        open(args.out_path, "w") as fout:
        for l in fin:
            phns = l.strip().split()
            phn39s = [mapping[phn] for phn in phns]
            fout.write(" ".join(phn39s) + "\n")

if __name__ == "__main__":
    main()
