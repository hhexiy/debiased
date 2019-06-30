import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--nli-file', help='original data in SNLI/MNLI format')
parser.add_argument('--output')
parser.add_argument('--format', choices=['SNLI', 'MNLI'], default='SNLI')
args = parser.parse_args()

with open(args.nli_file) as fin, open(args.output, 'w') as fout:
    reader = csv.DictReader(fin, delimiter='\t')
    if args.format == 'SNLI':
        fieldnames = ['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'captionID', 'pairID', 'label1', 'label2', 'label3', 'label4', 'label5']
    elif args.format == 'MNLI':
        fieldnames = ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label']
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for row in reader:
        new_row = dict(row)
        new_row['sentence1_binary_parse'] = row['sentence2_binary_parse']
        new_row['sentence2_binary_parse'] = row['sentence1_binary_parse']
        new_row['sentence1_parse'] = row['sentence2_parse']
        new_row['sentence2_parse'] = row['sentence1_parse']
        new_row['sentence1'] = row['sentence2']
        new_row['sentence2'] = row['sentence1']
        if row.get('gold_label', None) in ('entailment', 'neutral'):
            new_row['gold_label'] = 'non-contradiction'
        else:
            new_row['gold_label'] = row.get('gold_label', None)
        writer.writerow(new_row)

