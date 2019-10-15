import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knn-dir', nargs='+')

if __name__ == '__main__':
    args = parse_args()
    main(args)
