import argparse
import json
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file')
    parser.add_argument('--tsv-file')
    return parser.parse_args()

def main(args):
    data = [json.loads(line) for line in open(args.json_file).readlines()]
    with open(args.tsv_file, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t')
        header = ['index', 'captionID', 'pairID', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label']
        writer.writerow(header)
        for d in data:
            annotator_labels = d['annotator_labels']
            if len(annotator_labels) < 5:
                annotator_labels += [None] * (5 - len(annotator_labels))
            try:
                row = [d.get('index'),
                       d.get('captionID'),
                       d.get('pairID'),
                       d.get('sentence1_binary_parse'),
                       d.get('sentence2_binary_parse'),
                       d.get('sentence1_parse'),
                       d.get('sentence2_parse'),
                       d.get('sentence1'),
                       d.get('sentence2'),
                       annotator_labels[0],
                       annotator_labels[1],
                       annotator_labels[2],
                       annotator_labels[3],
                       annotator_labels[4],
                       d.get('gold_label')
                      ]
            except IndexError:
                print(annotator_labels)
                print(d)
            writer.writerow(row)

if __name__ == '__main__':
    args = parse_args()
    main(args)
