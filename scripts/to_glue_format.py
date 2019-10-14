"""Convert QQP data from Wang et al. (https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing) to GLUE format.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--source', choices=['wang', 'paws', 'paws-wiki', 'sick'])
    args = parser.parse_args()
    return args

def wang_to_glue(line):
    label, q1, q2, id_ = line.split('\t')
    return id_, 'null', 'null', q1, q2, label

def paws_to_glue(line):
    id_, q1, q2, label = line.split('\t')
    return id_, 'null', 'null', q1, q2, label

def sick_to_glue(line):
    id_, q1, q2, label = line.split('\t')[:4]
    return id_, 'null', 'null', q1, q2, label.lower()

def main(args):
    if args.source in ('wang', 'paws-wiki'):
        splits = ('train', 'dev', 'test')
    elif args.source == 'sick':
        splits = ('test',)
    else:
        splits = ('dev_and_test', 'train')
    for split in splits:
        with open('{}/{}.tsv'.format(args.in_dir, split)) as fin, \
             open('{}/{}.tsv'.format(args.out_dir, split), 'w') as fout:
            fout.write('\t'.join(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']) + '\n')
            for line in fin:
                if args.source == 'wang':
                    ss = wang_to_glue(line.strip())
                elif args.source == 'sick':
                    ss = sick_to_glue(line.strip())
                else:
                    ss = paws_to_glue(line.strip())
                if ss is not None:
                    fout.write('\t'.join(ss) + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
