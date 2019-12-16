import argparse
import os
from random import shuffle
import numpy as np
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', nargs='+')
    parser.add_argument('--out-dir')
    parser.add_argument('--task', choices=['nli'])
    parser.add_argument('--criteria', choices=['length'])
    parser.add_argument('--num-finetune', type=int, default=0)
    args = parser.parse_args()
    return args

def read_data(path):
    examples = []
    with open(path) as fin:
        header = fin.readline().strip().split('\t')
        header = {name: ind for ind, name in enumerate(header)}
        for line in fin:
            examples.append(line.strip().split('\t'))
    return examples, header

def read_all_data(paths):
    examples = []
    header = None
    sizes =[]
    for path in paths:
        _examples, _header = read_data(path)
        examples.extend(_examples)
        header = _header
        sizes.append(len(_examples))
    return examples, header, sizes

def _get_nli_length(example, header):
    s1 = example[header['sentence1']]
    s2 = example[header['sentence2']]
    l = max(len(s1.split()), len(s2.split()))
    return l

def get_length(examples, header, task):
    if task == 'nli':
        examples = [(_get_nli_length(e, header), e) for e in examples]
    return examples

def add_attribute(examples, header, task, criteria):
    if criteria == 'length':
        examples = get_length(examples, header, task)
    return examples

def split_examples(examples, criteria, train_size):
    """
    split_sizes: list[int] [train_size, dev_size, test_size]
    """
    if criteria == 'length':
        examples = sorted(examples, key=lambda x: x[0])

    dev_size = round((len(examples) - train_size) / 2)
    test_size = len(examples) - train_size - dev_size
    split_sizes = [train_size, dev_size, test_size]
    cum_split_sizes = np.cumsum(split_sizes)

    train_dev_examples = examples[:cum_split_sizes[1]]
    test_examples = examples[cum_split_sizes[1]:cum_split_sizes[2]]
    # mix train and dev examples
    shuffle(train_dev_examples)
    train_examples = train_dev_examples[:cum_split_sizes[0]]
    dev_examples = train_dev_examples[cum_split_sizes[0]:cum_split_sizes[1]]

    for split, examples in zip(('train', 'dev', 'test'), (train_examples, dev_examples, test_examples)):
        if criteria == 'length':
            print('{}: {}'.format(split, stats.mode([e[0] for e in examples])))

    return [e[1] for e in train_examples], \
           [e[1] for e in dev_examples], \
           [e[1] for e in test_examples]

def write_examples(examples, header, path):
    # Only write gold label because dev and train have different number of labels
    header = [h[0] for h in sorted(header.items(), key=lambda x: x[1]) if not (h[0].startswith('label'))]
    with open(path, 'w') as fout:
        fout.write('\t'.join(header) + '\n')
        for e in examples:
            fout.write('\t'.join(e) + '\n')


def main(args):
    examples, header, sizes = read_all_data(args.data_paths)
    examples = add_attribute(examples, header, args.task, args.criteria)
    train_size = max(sizes)
    train_examples, dev_examples, test_examples = split_examples(examples, args.criteria, train_size)
    write_examples(train_examples, header, os.path.join(args.out_dir, 'train.tsv'))
    write_examples(dev_examples, header, os.path.join(args.out_dir, 'dev.tsv'))
    write_examples(test_examples, header, os.path.join(args.out_dir, 'test.tsv'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
