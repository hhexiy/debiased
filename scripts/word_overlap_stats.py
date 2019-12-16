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
    ne_sims = []
    e_sims = []
    e_subset = 0
    ne_subset = 0
    if args.output_file:
        fout = open(args.output_file, 'w')
    else:
        fout = None
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                if fout:
                    fout.write(line)
                continue
            ss = line.strip().split('\t')
            s1 = re.sub(r'[()]', '', ss[4]).strip().split()
            s2 = re.sub(r'[()]', '', ss[5]).strip().split()
            s1_words = set([w.lower() for w in s1])
            s2_words = set([w.lower() for w in s2])
            subset = True
            for w in s2_words:
                if not w in s1_words:
                    subset = False
            jaccard_sim = len(s1_words.intersection(s2_words)) / len(s1_words.union(s2_words)) * 1.
            label = ss[-1]
            if label != 'entailment':
                ne_sims.append(jaccard_sim)
                ne_subset += int(subset)
            else:
                e_sims.append(jaccard_sim)
                e_subset += int(subset)
            if fout and not (subset and label != 'entailment'):
                fout.write(line)

    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1]
    print('non-entailment')
    print(np.mean(ne_sims))
    print(np.histogram(ne_sims, bins=buckets))
    print(ne_subset)
    print('entailment')
    print(np.mean(e_sims))
    print(np.histogram(e_sims, bins=buckets))
    print(e_subset)

if __name__ == '__main__':
    args = parse_args()
    main(args)
