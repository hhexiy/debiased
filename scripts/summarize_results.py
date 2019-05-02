import argparse
import glob
import json
import shutil
import os
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    parser.add_argument('--test', action='store_true', help='parse test log')
    args = parser.parse_args()
    return args

def get_model_config(model_path):
    res = json.load(open('{}/report.json'.format(model_path)))['config']
    return res

def parse_file(path, test=False):
    #print('parsing {}'.format(path))
    try:
        res = json.load(open(path))
        config = res['config']
        model_config = get_model_config(config['init_from'])
        test_data = '{}-{}'.format(config['task_name'], config['test_split'])
        test_split = config['test_split']
        train_data = '{}-{}'.format(model_config['task_name'], model_config['test_split'])
        model_path = config['init_from'].split('/')
        model_path = '/'.join(model_path[:-1] + [model_path[-1][:5]])

        model_cheat = float(model_config['cheat'])
        test_cheat = float(config['cheat'])
        wdrop = float(model_config.get('word_dropout', 0))
        model = model_config.get('model_type', 'bert') or 'bert'
        superficial = model_config['superficial'] if model_config['superficial'] else '-1'
        additive = len(model_config['additive']) if model_config['additive'] else 0
        last = int(config['use_last'])
        metrics = res['test'][test_split]
        if additive == 0:
            acc = metrics['accuracy']
            additive = '0'
        else:
            acc = metrics['last_accuracy']
            prev_models = []
            for prev in model_config['additive']:
                prev_config = get_model_config(prev)
                if prev_config['superficial'] == 'handcrafted':
                    prev_models.append('hand')
                elif prev_config['superficial']:
                    prev_models.append('sup')
                else:
                    prev_models.append('cbow')
            additive = ','.join(prev_models)
    except Exception as e:
        traceback.print_exc()
        print(os.path.dirname(path))
        #import sys; sys.exit()
        #shutil.rmtree(os.path.dirname(path))
        return 'failed'
    report = {
            'train_data': train_data,
            'test_data': test_data,
            'last': last,
            'mch': model_cheat,
            'tch': test_cheat,
            'sup': superficial,
            'add': additive,
            'wdrop': wdrop,
            'acc': acc,
            'path': model_path,
           }
    constraints = {
            #lambda r: r['mch'] == -1,
            #lambda r: r['tch'] == -1,
            #lambda r: r['sup'] == 0,
            #lambda r: r['add'] == '0',
            #lambda r: r['wdrop'] == 0,
            #lambda r: r['last'] == 0,
            lambda r: not r['test_data'].endswith('mismatched'),
            }
    for c in constraints:
        if not c(report):
            return 'filtered'
    return report

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f, test=args.test) for f in files]
    failed_paths = [f for r, f in zip(all_res, files) if r == 'failed']
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

    all_res = [r for r in all_res if not r in ('failed', 'filtered')]

    columns = [
               ('train_data', 20, 's'),
               ('test_data', 40, 's'),
               ('tch', 6, '.1f'),
               ('mch', 6, '.1f'),
               ('last', 5, 'd'),
               ('sup', 5, 's'),
               ('add', 10, 's'),
               ('wdrop', 10, '.1f'),
               ('acc', 10, '.3f'),
               ('path', 10, 's'),
              ]
    if len(all_res) == 0:
        print('no results found')
        return
    if 'last_acc' in all_res[0]:
        columns.append(('last_val_acc', 10, '.2f'))
    if 'prev_acc' in all_res[0]:
        columns.append(('prev_val_acc', 10, '.2f'))
    header = ''.join(['{{:<{w}s}}'.format(w=width)
                      for _, width, _ in columns])
    header = header.format(*[c[0] for c in columns])
    row_format = ''.join(['{{{c}:<{w}{f}}}'.format(c=name, w=width, f=form)
                          for name, width, form in columns])
    all_res = sorted(all_res, key=lambda x: [x[c[0]] for c in columns])

    print(header)
    for res in all_res:
        print(row_format.format(**res))

if __name__ == '__main__':
    args = parse_args()
    main(args)
