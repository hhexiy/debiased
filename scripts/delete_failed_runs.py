import argparse
import glob
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    args = parser.parse_args()
    return args

def main(args):
    failed_runs = []
    for path in args.runs_dir:
        dirs = glob.glob(path)
        for d in dirs:
            if not os.path.exists('{}/report.json'.format(d)):
                failed_runs.append(d)
    for d in failed_runs:
        print(d)
    r = input('Remove the above directories? [Y/N]')
    if str(r) == 'Y':
        for d in failed_runs:
            shutil.rmtree(d)

if __name__ == '__main__':
    args = parse_args()
    main(args)
