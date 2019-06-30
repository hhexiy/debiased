import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--hans-data')
parser.add_argument('--outdir')
args = parser.parse_args()

with open(args.hans_data) as fin:
    reader = csv.DictReader(fin, delimiter='\t')
    example_by_heu = defaultdict(list)
    for row in reader:
        example_by_heu[row['heuristic']].append(row)

for h, examples in example_by_heu.items():
    with open('{}/{}.tsv'.format(args.outdir, h), 'w') as fout:
        writer = csv.writer(fout, delimiter='\t')
        header = ['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'pairID']
        writer.writerow(header)
        for e in examples:
            writer.writerow([e[k] for k in header])

