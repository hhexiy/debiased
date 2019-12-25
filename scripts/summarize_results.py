import argparse
import sys
import glob
import json
import shutil
import os
import json
import traceback
import csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.tokenizer import BasicTokenizer

tokenizer = BasicTokenizer(do_lower_case=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    parser.add_argument('--output-json')
    parser.add_argument('--combine-hans', action='store_true', default=False)
    parser.add_argument('--error-analysis', default=None)
    parser.add_argument('--group-results', default=None)
    parser.add_argument('--aggregate-seeds', default=None)
    args = parser.parse_args()
    return args

def get_model_config(model_path):
    res = json.load(open('{}/report.json'.format(model_path)))['config']
    return res

def get_para_group_accuracy(path):
    print('paraphrase', path)
    pred_file = os.path.join('{}/predictions.tsv'.format(os.path.dirname(path)))
    with open(pred_file) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        results = []
        for row in reader:
            if int(row['label']) == 1:
                continue
            p_words = set(tokenizer.tokenize(row['premise']))
            h_words = set(tokenizer.tokenize(row['hypothesis']))
            overlap = len(p_words.intersection(h_words)) / len(p_words.union(h_words))
            #overlap = int(set(p_words) == set(h_words))
            results.append((overlap, int(row['correct'] == 'True')))
        #bins = [0, 0.2, 0.4, 0.6, 0.8, 1.1]
        bins = [0, 1, 1.1]
        bin_ids = np.digitize([r[0] for r in results], bins=bins) - 1
        n_groups = len(bins) - 1
        group_accs = [None] * n_groups
        for i in range(n_groups):
            _group_results = [r[1] for bin_id, r in zip(bin_ids, results) if int(bin_id) == i]
            if len(_group_results) > 0:
                group_accs[i] = (sum(_group_results) / len(_group_results), len(_group_results))
            else:
                print('skipping group {}-{}'.format(bins[i], bins[i+1]))
        return group_accs

def get_nli_group_accuracy(path):
    pred_file = os.path.join('{}/predictions.tsv'.format(os.path.dirname(path)))
    with open(pred_file) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        results = []
        for row in reader:
            if row['label'] == 'entailment':
                continue
            p_words = tokenizer.tokenize(row['premise'])
            h_words = tokenizer.tokenize(row['hypothesis'])
            overlap = len([w for w in h_words if w in p_words]) / float(len(h_words))
            results.append((overlap, int(row['correct'] == 'True')))
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.1]
        bin_ids = np.digitize([r[0] for r in results], bins=bins) - 1
        n_groups = len(bins) - 1
        group_accs = [None] * n_groups
        for i in range(n_groups):
            _group_results = [r[1] for bin_id, r in zip(bin_ids, results) if int(bin_id) == i]
            if len(_group_results) > 0:
                group_accs[i] = (sum(_group_results) / len(_group_results), len(_group_results))
            else:
                print('skipping group {}-{}'.format(bins[i], bins[i+1]))
        return group_accs

def group_accuracy(paths, group='nli'):
    all_groups = []
    if group == 'nli':
        get_acc = get_nli_group_accuracy
    else:
        get_acc = get_para_group_accuracy
    for path in paths:
        all_groups.append(get_acc(path))
    n_groups = len(all_groups[0])
    for i in range(n_groups):
        accs = [g[i][0] for g in all_groups if g[i] is not None]
        sizes = [g[i][1] for g in all_groups if g[i] is not None]
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        avg_size = np.mean(sizes)
        print('{:3d}{:>10.4f}{:>10.4f}{:>10.2f}'.format(i, avg_acc, std_acc, avg_size))


def analyze(path, data):
    pred_file = os.path.join('{}/predictions.tsv'.format(os.path.dirname(path)))
    with open(pred_file) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        preds, labels = [], []
        for row in reader:
            if data == 'hans':
                preds.append('non-entailment' if row['pred'] != 'entailment' else 'entailment')
                labels.append(row['label'])
            elif data == 'swap':
                preds.append('non-contradiction' if row['pred'] != 'contradiction' else 'contradiction')
                labels.append(row['label'])
            else:
                preds.append(row['pred'])
                labels.append(row['label'])
    report = classification_report(labels, preds, output_dict=True)
    return report

def parse_file(path, error_analysis):
    #print('parsing {}'.format(path))
    try:
        res = json.load(open(path))
        config = res['config']
        model_config = get_model_config(config['init_from'])
        test_data = '{}-{}'.format(config['task_name'], config['test_split'])
        test_split = config['test_split']
        train_data = '{}-{}'.format(model_config['task_name'], model_config['test_split'])
        model_path = config['init_from'].split('/')
        #model_path = '/'.join(model_path[:-1] + [model_path[-1][:5]])
        model_path = '/'.join(model_path[:-1] + [model_path[-1]])
        model = model_config['model_type']
        model_name = model_config['model_name']
        model_params = model_config.get('model_params')
        if model_params is None:
            model_params = 'pretrained'
        else:
            model_params = model_params.split('/')[-1]
        seed = model_config['seed']

        model_cheat = float(model_config['cheat'])
        test_cheat = float(config['cheat'])
        rm_cheat = float(model_config.get('remove_cheat', False))
        rm_overlap = float(model_config.get('remove_overlap', 0))
        rm_random = float(model_config.get('remove_random', 0))
        wdrop = float(model_config.get('word_dropout', 0))
        aug = int(model_config.get('augment_by_epoch', 0))
        wmask = float(model_config.get('word_mask', 0))
        model = model_config.get('model_type', 'bert') or 'bert'
        superficial = model_config['superficial'] if model_config['superficial'] else '-1'
        additive = len(model_config['additive']) if model_config['additive'] else 0
        last = int(config['use_last'])
        metrics = res['test'][test_split]
        if test_data.startswith('MNLI-hans') or test_data.startswith('SNLI-swap') or test_data.startswith('MNLI-swap'):
            metric_name = 'mapped-accuracy'
        else:
            metric_name = 'accuracy'
        if test_data.startswith('QQP'):
            f1 = metrics['f1']
        else:
            f1 = -1.
        if additive == 0:
            acc = metrics[metric_name]
            additive = '0'
            if model_config['superficial'] is True or model_config['superficial'] == 'hypo':
                model = 'hypo'
            elif model_config['superficial'] == 'handcrafted':
                model = 'hand'
        else:
            if config['additive_mode'] == 'all':
                acc = metrics['last_{}'.format(metric_name)]
            elif config['additive_mode'] == 'last':
                acc = metrics['{}'.format(metric_name)]
            else:
                raise ValueError
            prev_models = []
            for prev in model_config['additive']:
                prev_config = get_model_config(prev)
                if prev_config['superficial'] == 'handcrafted':
                    prev_models.append('hand')
                elif prev_config['superficial']:
                    prev_models.append('hypo')
                else:
                    prev_models.append('cbow')
            additive = ','.join(prev_models)
    except Exception as e:
        traceback.print_exc()
        print(os.path.dirname(path))
        #import sys; sys.exit()
        #shutil.rmtree(os.path.dirname(path))
        return {
                'status': 'failed',
                'eval_path': path,
               }
    remove = int(model_config.get('remove', False))
    if remove == 1:
        assert not config['remove']
    report = {
            'status': 'success',
            'seed': seed,
            'train_data': train_data,
            'test_data': test_data,
            'model_params': model_params,
            'last': last,
            'mch': model_cheat,
            'tch': test_cheat,
            'rm_ch': rm_cheat,
            'sup': superficial,
            'add': additive,
            'rm': remove,
            'rm_overlap': rm_overlap,
            'rm_random': rm_random,
            'wdrop': wdrop,
            'wmask': wmask,
            'aug': aug,
            'model': model.upper(),
            'model_name': model_name,
            'acc': acc,
            'f1': f1,
            'model_path': model_path,
            'eval_path': path,
           }
    constraints = {
            lambda r: r['model_params'] == 'pretrained',
            lambda r: r['model_name'] != 'openwebtext_book_corpus_wiki_en_uncased',
            #lambda r: r['tch'] == 0,
            #lambda r: r['sup'] == 0,
            #lambda r: r['add'] in ('hand', 'hypo', 'cbow', '0'),
            #lambda r: r['add'] != 'hypo,cbow',
            lambda r: r['wdrop'] in (0,),
            lambda r: r['wmask'] in (0,),
            #lambda r: r['rm'] in (0,),
            #lambda r: r['test_data'].startswith('MNLI-hans'),
            #lambda r: r['train_data'] == 'QQP-wang-dev',
            #lambda r: r['test_data'] == 'QQP-wang',
            lambda r: r['train_data'] == 'MNLI-dev_matched',
            #lambda r: r['test_data'] == 'MNLI-hans-lexical_overlap',
            #lambda r: r['train_data'] == 'SNLI',
            #lambda r: not r['test_data'].endswith('mismatched'),
            #lambda r: r['model'] in ('BERT',),
            lambda r: r['rm_overlap'] == 0.064 and r['rm_random'] == 0,
            #lambda r: r['rm_overlap'] != 0,
            #lambda r: r['model'] in ('ESIM','HYPO', 'CBOW', 'HAND'),
            #lambda r: r['model'] in ('HYPO', 'CBOW', 'HAND'),
            }
    for c in constraints:
        if not c(report):
            return {
                    'status': 'filtered',
                    'eval_path': path,
                   }
    if error_analysis is not None:
        try:
            acc_report = analyze(path, error_analysis)
            if error_analysis == 'hans':
                report.update({
                    'ent': acc_report['entailment']['f1-score'],
                    'n-ent': acc_report['non-entailment']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
            elif error_analysis == 'swap':
                report.update({
                    'con': acc_report['contradiction']['f1-score'],
                    'n-con': acc_report['non-contradiction']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
            elif error_analysis == 'qqp':
                report.update({
                    'f1': acc_report['macro avg']['f1-score']
                    })
            else:
                report.update({
                    'ent': acc_report['entailment']['f1-score'],
                    'con': acc_report['contradiction']['f1-score'],
                    'neu': acc_report['neutral']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
        except Exception as e:
            traceback.print_exc()
            print(os.path.dirname(path))
            return {
                    'status': 'failed',
                    'eval_path': path,
                   }
    return report

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f, args.error_analysis) for f in files]
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

    if args.group_results in ('nli', 'para'):
        group_accuracy([r['eval_path'] for r in all_res], args.group_results)

    columns = [
               ('test_data', 30, 's'),
               ('model', 10, 's'),
               #('train_data', 30, 's'),
               #('tch', 6, '.1f'),
               #('mch', 6, '.1f'),
               #('rm_ch', 6, '.1f'),
               #('sup', 5, 's'),
               #('wdrop', 6, '.1f'),
               #('wmask', 6, '.1f'),
               ('rm_overlap', 10, '.3f'),
               ('rm_random', 10, '.3f'),
               #('aug', 6, 'd'),
               #('model_name', 40, 's'),
               #('model_params', 45, 's'),
               ('seed', 7, 'd'),
               #('rm', 5, 'd'),
               #('add', 7, 's'),
               ('acc', 10, '.3f'),
               #('model_path', 10, 's'),
               #('eval_path', 10, 's'),
              ]
    if args.error_analysis == 'hans':
        columns.extend([
               ('ent', 10, '.3f'),
               ('n-ent', 10, '.3f'),
               ('avg', 10, '.3f'),
            ])
    elif args.error_analysis == 'swap':
        columns.extend([
               ('con', 10, '.3f'),
               ('n-con', 10, '.3f'),
            ])
    elif args.error_analysis == 'qqp':
        columns.extend([
                ('f1', 10, '.3f'),
            ])
    elif args.error_analysis is not None :
        columns.extend([
               ('ent', 10, '.3f'),
               ('con', 10, '.3f'),
               ('neu', 10, '.3f'),
               ('avg', 10, '.3f'),
            ])
    #columns.append(('eval_path', 10, 's'))
    if len(all_res) == 0:
        print('no results found')
        return
    #if 'last_acc' in all_res[0]:
    #    columns.append(('last_val_acc', 10, '.2f'))
    #if 'prev_acc' in all_res[0]:
    #    columns.append(('prev_val_acc', 10, '.2f'))
    all_res = sorted(all_res, key=lambda x: [x[c[0]] for c in columns])

    column_names = [c[0] for c in columns]

    new_res = []
    if args.combine_hans:
        res_groups = defaultdict(dict)
        hans_split = {}
        invariant_cols = [c for c in column_names if not c in ('test_data', 'acc', 'eval_path')]
        for i, res in enumerate(all_res):
            val = [res[c] for c in invariant_cols]
            res_groups[str(val)][res['test_data']] = res
        for k, g in res_groups.items():
            _r = eval(k)
            assert len(g) == 3
            avg_acc = np.mean([r['acc'] for r in g.values()])
            r = list(g.values())[0]
            r['acc'] = avg_acc
            r['test_data'] = 'HANS-combined'
            new_res.append(r)
        all_res = new_res

    # Aggregate over seeds assuming that the records are sorted already
    # such that records with different seeds are grouped together
    # NOTE: records in the same group must have same values left to the
    # 'seed' column.
    if args.aggregate_seeds and 'seed' in column_names:
        seed_idx = column_names.index('seed')
        seed_invariant_cols = column_names[:seed_idx]
        agg_cols = ['acc', 'ent', 'n-ent', 'f1']
        res_groups = []
        group_seeds = []
        for i, res in enumerate(all_res):
            val = [res[c] for c in seed_invariant_cols]
            prev_val = None if i == 0 else [all_res[i-1][c] for c in seed_invariant_cols]
            if val == prev_val:
                # NOTE: sometimes there are repeated seeds - don't count them
                if res['seed'] in group_seeds:
                    continue
                res_groups[-1].append(res)
                group_seeds.append(res['seed'])
            else:
                group_seeds = []
                res_groups.append([res])
                group_seeds.append(res['seed'])
        new_res = []
        for res_group in res_groups:
            agg_res = {}
            for col in column_names:
                if col in agg_cols:
                    vals = [r[col] for r in res_group]
                    val_mean = np.mean(vals)
                    val_std = np.std(vals)
                    agg_res[col] = '{:.3f}/{:.3f}'.format(val_mean, val_std)
                    agg_res[col+'-mean'] = val_mean
                    agg_res[col+'-std'] = val_std
                else:
                    if col == 'seed':
                        agg_res['#seed'] = len(res_group)
                    else:
                        agg_res[col] = res_group[0][col]
            new_res.append(agg_res)
        all_res = new_res
        # Fix column formatting
        for i, col in enumerate(columns):
            if col[0] in agg_cols:
                columns[i] = (col[0], 15, 's')
            elif col[0] == 'seed':
                columns[i] = ('#seed', col[1], col[2])

    header = ''.join(['{{:<{w}s}}'.format(w=width)
                      for _, width, _ in columns])
    header = header.format(*[c[0] for c in columns])
    row_format = ''.join(['{{{c}:<{w}{f}}}'.format(c=name, w=width, f=form)
                          for name, width, form in columns])

    #duplicated_paths = []
    #for i, r in enumerate(all_res):
    #    if i > 0 and r['model_path'] == all_res[i-1]['model_path']:
    #        f = r['eval_path']
    #        duplicated_paths.append(f)
    #        print(f)
    #ans = input('remove duplicated paths? [Y/N]')
    #if ans == 'Y':
    #    for f in duplicated_paths:
    #        shutil.rmtree(os.path.dirname(f))
    #import sys; sys.exit()

    #to_delete = [r['eval_path'] for r in all_res]
    #for p in to_delete:
    #    print(p)
    #ans = input('remove selected paths? [Y/N]')
    #if ans == 'Y':
    #    for f in to_delete:
    #        shutil.rmtree(os.path.dirname(f))
    #    import sys; sys.exit()


    print(header)
    for res in all_res:
        print(row_format.format(**res))

    if args.output_json:
        with open(args.output_json, 'w') as fout:
            json.dump(all_res, fout, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
