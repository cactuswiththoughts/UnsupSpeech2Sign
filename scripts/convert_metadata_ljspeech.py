import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("tsv_path")
    parser.add_argument("out_path")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    text_dict = dict() 
    with open(args.in_path, "r") as f_in,\
        open(args.tsv_path, "r") as f_tsv,\
        open(args.out_path, "w") as f_out:
        for line in f_in:
            utt_id, text = line.strip().split('|')[:2]
            filtered = "".join(
                [c.upper() for c in text 
                if "A" <= c.upper() <= "Z" or c == " "]
            )
            text_dict[utt_id] = filtered
        
        lines = f_tsv.read().strip().split("\n")
        _ = lines.pop(0)
        for line in lines:
            fn = Path(line.split("\t")[0])
            utt_id = fn.stem
            text = text_dict[utt_id]
            f_out.write(text+"\n")

if __name__ == "__main__":
    main()
