import os

from text_mining.text_mining import *

# A python Script to Run all the classification experiments

def make_args_parser():

    parser = argparse.ArgumentParser(description='Text mining library')
    parser.add_argument('--datasets', default='datasets',
                        help='Define the relative path of the dataset directory')
    parser.add_argument('--outputs', default='outputs',
                        help='Define the relative path of the output directory')
    return parser.parse_args()

def print_config(args):
    print("Running Experiments with the following configuration")
    arg_map = vars(args)
    for key in arg_map:
        print('\t', key, '->',  arg_map[key])

def main():

    args = make_args_parser()
    print_config(args)
    

    # Feeding the preprocessed_csv to avoid unnecessary overhead

    with open(path + 'EvaluationMetric_10fold.csv', 'w') as f:

		f.write("Statistic-Measure\tSvm-(BoW)\tRandom-Forest-(BoW)\tSVM-(SVD)\tRandom-Forest-(SVD)\tSVM-(W2V)\tRandom-Forest-(W2V)\tMy-Method\n")
		f.write("Accuracy\n")
		f.write("Precision\n")
		f.write('Recall\n')

	exit(1)
	# Experiment 1 : Support Vector Machines - BoW
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="BoW", kfold=True)
	exp.run()

	# Experiment 2 : Random Forest - BoW
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="BoW", kfold=True)
	exp.run()

	# Experiment 3 :  Support Vector Machines - SVD
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="SVD", kfold=True)
	exp.run()

	# Experiment 4 :  Random Forest - SVD
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="SVD", kfold=True)
	exp.run()
    
	# Experiment 5 :  Support Vector Machines - W2V
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="W2V", kfold=True)
	exp.run()

	# Experiment 6 :  Random Forest - W2V
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="W2V", kfold=True)
	exp.run()

	# Experiment 6 :  Random Forest - W2V
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="W2V", kfold=True)
	exp.run()


if __name__ == '__main__':
    main()


