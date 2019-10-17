import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from joblib import dump, load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knn-dir', nargs='+')
    args = parser.parse_args()
    return args

def main(args):
    path = os.path.join(args.knn_dir[0], 'knns.joblib')
    knns = load(path)
    print(knns.keys())
    print(len(knns.keys()))
    fig, axs = plt.subplots(1, len(knns.keys()))
    for i, (label_name, _knns) in enumerate(knns.items()):
        max_dists = np.max(_knns[0], axis=1)
        axs[i].hist(max_dists)
        axs[i].set_title(label_name)
    plt.savefig('tmp.pdf')



if __name__ == '__main__':
    args = parse_args()
    main(args)
