import argparse
import csv
from sklearn.metrics import classification_report, confusion_matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-file')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.pred_file) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        preds, labels = [], []
        for row in reader:
            preds.append('non-entailment' if row['pred'] != 'entailment' else 'entailment')
            #preds.append('non-contradiction' if row['pred'] != 'contradiction' else 'contradiction')
            labels.append(row['label'])

    report = classification_report(labels, preds)
    print(report)

    conf = confusion_matrix(labels, preds)
    labels = sorted(list(set(labels)))
    print_cm(conf, labels=labels)


if __name__ == '__main__':
    args = parse_args()
    main(args)
