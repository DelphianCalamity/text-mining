import argparse

from text_mining.text_mining import *


def make_args_parser():

    parser = argparse.ArgumentParser(description='Text mining library')
    parser.add_argument('--datasets', default='datasets',
                        help='Define the relative path of the dataset directory')
    parser.add_argument('--outputs', default='outputs',
                        help='Define the relative path of the output directory')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess articles')
    parser.add_argument('--wordclouds', action='store_true',
                        help='Generate word clouds for each category')
    parser.add_argument('--duplicates', dest='threshold', type=float, default=None,
                        help='Identify similar documents within above a certain theta')
    parser.add_argument('--classification', choices=['SVM', 'RF'],
                        help='Runs default classifiers with 10-fold cross validation: Random Forest and SVM')
    parser.add_argument('--features', choices=['BoW', 'SVD', 'W2V'], default = None,
                        help='Define features')
    parser.add_argument('--kfold', action='store_true',
                        help='Evaluate and report the performance of each method using 10-fold Cross Validation')

    return parser.parse_args()


def print_config(args):
    print("Running with the following configuration")
    arg_map = vars(args)
    for key in arg_map:
        print('\t', key, '->',  arg_map[key])


def main():

    args = make_args_parser()
    print_config(args)
    
    text = TextMining(args.datasets, args.outputs, args.preprocess, args.wordclouds, 
            args.threshold, args.classification, args.features, args.kfold)
    
    text.run()


if __name__ == '__main__':
    main()
