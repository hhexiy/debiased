"""Convert QQP data from Wang et al. (https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing) to GLUE format.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    return args

def wang_to_glue(line):
    label, q1, q2, id_ = line.split('\t')
    return id_, 'null', 'null', q1, q2, label

def main(args):
    for split in ('train', 'dev', 'test'):
        with open('{}/{}.tsv'.format(args.in_dir, split)) as fin, \
             open('{}/{}.tsv'.format(args.out_dir, split), 'w') as fout:
            fout.write('\t'.join(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']) + '\n')
            for line in fin:
                ss = wang_to_glue(line.strip())
                fout.write('\t'.join(ss) + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
