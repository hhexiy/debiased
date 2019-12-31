import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
from scipy import stats
import nltk
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    return args

def swapped_word_tag(s1, s2):
    tag1 = nltk.pos_tag(s1)
    for i, (w1, w2) in enumerate(zip(s1, s2)):
        if w1 != w2:
            return tag1[i][1]
    return None

def main(args):
    path = args.csv_file
    if args.output_file:
        fout = open(args.output_file, 'w')
    else:
        fout = None
    p_same_bow = 0
    np_same_bow = 0
    p_sim = []
    np_sim = []
    n_remove = 0
    swapped_tags = defaultdict(int)
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                if fout:
                    fout.write(line.strip() + '\tsim\n')
                continue
            ss = line.strip().split('\t')
            s1 = ss[3].lower().split()
            s2 = ss[4].lower().split()
            s1_words = set(s1)
            s2_words = set(s2)
            jaccard_sim = len(s1_words.intersection(s2_words)) / len(s2_words.union(s1_words))
            same_bow = s1_words == s2_words
            label = int(ss[-1])
            if label == 1:
                p_same_bow += int(same_bow)
                p_sim.append(jaccard_sim)
            else:
                np_same_bow += int(same_bow)
                np_sim.append(jaccard_sim)
            if jaccard_sim == 1 and label == 0:
                swapped_tag = swapped_word_tag(s1, s2)
                if swapped_tag:
                    swapped_tags[swapped_tag[:2]] += 1
            #if jaccard_sim == 1 and label != 1:
            #    print(line)
            if fout:
                fout.write(line.strip() + '\t{}\n'.format(jaccard_sim))
                #if not (ss[3].lower() == ss[4].lower() and label != 1):
                #    fout.write(line)
                #else:
                #    n_remove += 1
    print(n_remove)
    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.1]
    print('same BOW')
    print('paraphrase', p_same_bow)
    print('non-paraphrase', np_same_bow)
    print('jaccard sim')
    print('paraphrase', np.mean(p_sim))
    h, _ = np.histogram(p_sim, bins=buckets)
    print(h)
    print(h / np.sum(h))
    print('non-paraphrase', np.mean(np_sim))
    h, _ = np.histogram(np_sim, bins=buckets)
    print(h)
    print(h / np.sum(h))
    print('swapped tags')
    print(sorted(swapped_tags.items(), key=lambda x: x[1], reverse=True))

    sns.set(style="whitegrid")
    sns.set(font_scale=1.2)
    #g = sns.distplot(p_sim, label='paraphrase', kde=False)
    g = sns.distplot(p_sim + np_sim, label='paraphrase', kde=False, hist_kws={'alpha': 1})
    g = sns.distplot(np_sim, label='non-paraphrase', kde=False, hist_kws={'alpha': 1})
    g.legend()
    g.set(xlabel='Jaccard similarity')
    fig = g.get_figure()
    fig.tight_layout()
    fig.savefig('figures/qqp-similarity.pdf')

if __name__ == '__main__':
    args = parse_args()
    main(args)
