import argparse
import traceback
import shutil
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from joblib import dump, load
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    args = parser.parse_args()
    return args

def get_model_config(model_path):
    res = json.load(open('{}/report.json'.format(model_path)))['config']
    return res

def parse_file(path):
    try:
        res = json.load(open(path))
        config = res['config']
        knn_model_config = get_model_config(config['init_from'])
        test_data = '{}-{}'.format(config['task_name'], config['test_split'])
        test_split = config['test_split']
        train_data = '{}-{}'.format(knn_model_config['task_name'], knn_model_config['train_split'])
        knn_model_path = config['init_from'].split('/')
        knn_model_path = '/'.join(knn_model_path[:-1] + [knn_model_path[-1][:5]])
        model_config = get_model_config(knn_model_config['init_from'])
        model = model_config['model_type']
        knns = load(os.path.join(os.path.dirname(path), 'knns.joblib'))
    except Exception as e:
        traceback.print_exc()
        print(os.path.dirname(path))
        return {
                'status': 'failed',
                'eval_path': path,
               }
    report = {
            'status': 'success',
            'train_data': train_data,
            'test_data': test_data,
            'model': model.upper(),
            'model_path': knn_model_path,
            'eval_path': path,
            'knns': knns,
           }
    constraints = {
            }
    for c in constraints:
        if not c(report):
            return {
                    'status': 'filtered',
                    'eval_path': path,
                   }
    return report

def _plot_hist(res, name):
    num_cols = max([len(r['knns']) for r in res])
    num_rows = len(res)
    print(num_cols, num_rows)
    fig, axs = plt.subplots(num_rows, num_cols)
    for i, r in enumerate(res):
        for j, (label_name, _knns) in enumerate(r['knns'].items()):
            max_dists = np.max(_knns[0], axis=1)
            r['knn-{}'.format(label_name)] = float(np.mean(max_dists).reshape(-1))
            axs[i, j].hist(max_dists)
            axs[i, j].set_title('{}-{}'.format(r['test_data'], label_name))
    plt.savefig('{}.pdf'.format(name))

def plot_hists(all_res):
    groups = defaultdict(list)
    for r in all_res:
        knn_model = '{}-{}'.format(r['train_data'], r['model'])
        groups[knn_model].append(r)
    for group_name, res in groups.items():
        _plot_hist(res, group_name)

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f) for f in files]
    failed_paths = [r['eval_path'] for r in all_res if r['status'] == 'failed']
    if failed_paths:
        print('failed paths:')
        for f in failed_paths:
            print(f)
        ans = input('remove failed paths? [Y/N]')
        if ans == 'Y':
            for f in failed_paths:
                shutil.rmtree(os.path.dirname(f))
            print('removed {} dirs'.format(len(failed_paths)))
        else:
            print('ignore failed paths. continue')

    all_res = [r for r in all_res if r['status'] == 'success']
    plot_hists(all_res)

    knn_dists = [k for k in all_res[0].keys() if k.startswith('knn-')]
    columns = [
               ('train_data', 20, 's'),
               ('test_data', 30, 's'),
               ('model', 10, 's'),
              ]
    for k in knn_dists:
        columns.append((k, 10, '.2f'))

    all_res = sorted(all_res, key=lambda x: [x[c[0]] for c in columns])
    header = ''.join(['{{:<{w}s}}'.format(w=width)
                      for _, width, _ in columns])
    header = header.format(*[c[0] for c in columns])
    row_format = ''.join(['{{{c}:<{w}{f}}}'.format(c=name, w=width, f=form)
                          for name, width, form in columns])
    print(header)
    for res in all_res:
        print(row_format.format(**res))

if __name__ == '__main__':
    args = parse_args()
    main(args)
