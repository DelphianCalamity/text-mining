import argparse

from text_mining.text_mining import *

def make_args_parser():

    parser = argparse.ArgumentParser(description='Text mining library')
    parser.add_argument('--datasets', dest='datasets', default='datasets',
            help='Define the relative path of the dataset directory')
    parser.add_argument('--outputs', dest='outputs', default='outputs',
            help='Define the relative path of the output directory')
    parser.add_argument('--preprocess', dest='preprocess', default=False, 
            help='Preprocess articles')
    parser.add_argument('--wordclouds', dest='wordclouds', default=False, 
            help='Generate word clouds for each category')
    parser.add_argument('--duplicates', dest='duplicates', default=False, 
            help='Identify similar documents within above a certain theta')
    parser.add_argument('--classification', dest='classification', default=False, 
            help='Runs default classifiers with 10-fold cross validation: Random Forest and SVM')
    parser.add_argument('--features', dest='features', choices=['BoW', 'SVD', 'W2V'],
            default = None,
            help='Define features')

    return parser.parse_args()

def print_config(args):
    print "Running with the following configuration"
    arg_map = vars(args)
    for key in arg_map:
        print '\t', key, '->',  arg_map[key]

def main():

    args = make_args_parser()
    print_config(args)
    
    text = TextMining(args.datasets, args.outputs, args.preprocess,
            args.wordclouds, args.duplicates, args.classification, args.features)
    
    text.run()

if __name__ == '__main__':
    main()
