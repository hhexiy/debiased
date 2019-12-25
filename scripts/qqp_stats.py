import argparse
import re
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    return args

def main(args):
    path = args.csv_file
    if args.output_file:
        fout = open(args.output_file, 'w')
    else:
        fout = None
    p_same_bow = 0
    np_same_bow = 0
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                if fout:
                    fout.write(line)
                continue
            ss = line.strip().split('\t')
            s1 = ss[3].lower().split()
            s2 = ss[4].lower().split()
            s1_words = set(s1)
            s2_words = set(s2)
            same_bow = s1_words == s2_words
            label = int(ss[-1])
            if label == 1:
                p_same_bow += int(same_bow)
            else:
                np_same_bow += int(same_bow)
            if same_bow and fout:
                fout.write(line)
    print('same BOW')
    print('paraphrase', p_same_bow)
    print('non-paraphrase', np_same_bow)

if __name__ == '__main__':
    args = parse_args()
    main(args)
