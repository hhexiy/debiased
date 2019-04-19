import argparse
import glob
import json

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
    print('parsing {}'.format(path))
    res = json.load(open(path))
    data = res['config']['task_name']
    if test:
        config = get_model_config(res['config']['init_from'])
    else:
        config = res['config']
    cheat = float(config['cheat'])
    wdrop = float(config.get('word_dropout', 0))
    model = config.get('model_type', 'bert') or 'bert'
    superficial = int(config['superficial'])
    if test:
        metrics = res['test']['test']
    else:
        metrics = res['train']['best_val_results']
    acc = metrics['accuracy']
    report = {
            'data': data,
            'model': model,
            'cheat': cheat,
            'sup': superficial,
            'wdrop': wdrop,
            'acc': acc,
           }
    if config['additive']:
        report['last_acc'] = metrics['last_accuracy']
        report['prev_acc'] = metrics['prev_accuracy']
    return report

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f, test=args.test) for f in files]
    columns = [('data', 10, 's'),
               ('model', 7, 's'),
               ('sup', 10, 'd'),
               ('cheat', 10, '.1f'),
               ('wdrop', 10, '.1f'),
               ('acc', 10, '.2f'),
              ]
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
