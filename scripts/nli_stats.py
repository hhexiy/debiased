import argparse
import re
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file')
    parser.add_argument('--output-file')
    parser.add_argument('--remove-fraction', default=0, type=float)
    args = parser.parse_args()
    return args

def main(args):
    path = args.csv_file
    ne_sims = []
    ne_overlap = []
    e_sims = []
    e_overlap = []
    e_subset = 0
    ne_subset = 0
    if args.output_file:
        fout = open(args.output_file, 'w')
    else:
        fout = None
    examples = []
    n_total = 0
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                if fout:
                    fout.write(line)
                continue
            n_total += 1
            ss = line.strip().split('\t')
            s1 = re.sub(r'[()]', '', ss[4]).strip().split()
            s2 = re.sub(r'[()]', '', ss[5]).strip().split()
            s1_words = set([w.lower() for w in s1])
            s2_words = set([w.lower() for w in s2])

            s1_all_words = [w.lower() for w in s1]
            s2_all_words = [w.lower() for w in s2]
            overlap = len([w for w in s2_all_words if w in s1_all_words]) / float(len(s2_all_words))

            #s1 = ss[6].strip().split()
            #s2 = ss[7].strip().split()
            #s1_words = set([w.lower() for w in s1 if not w in ('.', ',', '!')])
            #s2_words = set([w.lower() for w in s2 if not w in ('.', ',', '!')])
            subset = True
            for w in s2_words:
                if not w in s1_words:
                    subset = False
            jaccard_sim = len(s1_words.intersection(s2_words)) / len(s1_words.union(s2_words)) * 1.
            label = ss[-1]
            if label != 'entailment':
                ne_sims.append(jaccard_sim)
                ne_overlap.append(overlap)
                ne_subset += int(subset)
                examples.append((overlap, line))
            else:
                e_sims.append(jaccard_sim)
                e_overlap.append(overlap)
                e_subset += int(subset)

    if fout:
        examples = sorted(examples, key=lambda x: x[0], reverse=True)
        n_remove = int(n_total * args.remove_fraction)
        examples = examples[n_remove:]
        for e in examples:
            fout.write(e[1])


    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.1]
    print('non-entailment')
    print(np.mean(ne_sims))
    print(np.histogram(ne_sims, bins=buckets))
    print(np.mean(ne_overlap))
    print(np.histogram(ne_overlap, bins=buckets))
    print(ne_subset)
    print(len([o for o in ne_overlap if o == 1.]))
    print('entailment')
    print(np.mean(e_sims))
    print(np.histogram(e_sims, bins=buckets))
    print(np.mean(e_overlap))
    print(np.histogram(e_overlap, bins=buckets))
    print(e_subset)

if __name__ == '__main__':
    args = parse_args()
    main(args)
