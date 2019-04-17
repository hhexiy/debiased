import argparse
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    args = parser.parse_args()
    return args

def parse_file(path):
    print('parsing {}'.format(path))
    res = json.load(open(path))
    data = res['config']['task_name']
    cheat = float(res['config']['cheat'])
    wdrop = float(res['config'].get('word_dropout', 0))
    model = res['config'].get('model_type', 'bert')
    superficial = int(res['config']['superficial'])
    val_acc = res['train']['best_val_results']['accuracy']
    report = {
            'data': data,
            'model': model,
            'cheat': cheat,
            'sup': superficial,
            'wdrop': wdrop,
            'val_acc': val_acc,
           }
    if res['config']['additive']:
        report['last_val_acc'] = res['train']['best_val_results']['last_accuracy']
        report['prev_val_acc'] = res['train']['best_val_results']['prev_accuracy']
    else:
        report['last_val_acc'] = -1
        report['prev_val_acc'] = -1
    return report

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f) for f in files]
    columns = [('data', 10, 's'),
               ('model', 7, 's'),
               ('sup', 10, 'd'),
               ('cheat', 10, '.1f'),
               ('wdrop', 10, '.1f'),
               ('val_acc', 10, '.2f'),
               ('last_val_acc', 10, '.2f'),
               ('prev_val_acc', 10, '.2f'),
              ]
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
