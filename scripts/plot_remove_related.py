import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--json-result')
args = parser.parse_args()

results = json.load(open(args.json_result))
tab_results = []
for r in results:
    if not r['model'] in ('BERT', 'ROBERTA', 'BERTL', 'ROBERTAL'):
        continue
    model_names = {
            'BERT': 'BERT-base',
            'ROBERTA': 'RoBERTa-base',
            'BERTL': 'BERT-large',
            'ROBERTAL': 'RoBERTa-large',
            }
    r['model'] = model_names[r['model']]
    if r['rm_overlap'] > 0:
        r['removed_fraction'] = r['rm_overlap'] * 100
        r['strategy'] = 'overlap'
        tab_results.append(r)
    elif r['rm_random'] > 0:
        r['removed_fraction'] = r['rm_random'] * 100
        r['strategy'] = 'random'
        tab_results.append(r)
    else:
        r['removed_fraction'] = 0
        r['strategy'] = 'overlap'
        tab_results.append(r)
        r = dict(r)
        r['strategy'] = 'random'
        tab_results.append(r)

df = pd.DataFrame(data=tab_results)

sns.set(style="whitegrid")
sns.set(font_scale=1.7)
g = sns.catplot(data=df, x='removed_fraction', y='acc', hue='strategy', col='model', kind='point')
g.set_axis_labels('% training data removed', 'accuracy (%)')
g.savefig('remove.pdf')
