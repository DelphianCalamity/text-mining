import os
import csv
import argparse
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


def PrintEvaluationFile(scores, path):

	with open(path + '/' + 'EvaluationMetric_10fold.csv', 'w') as f:
		
		scores = list(map(list, zip(*scores)))
		scores[0] = ["Accuracy"] + scores[0]
		scores[1] = ["Precision"] + scores[1]
		scores[2] = ["Recall"] + scores[2]

		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["Statistic-Measure","Svm-(BoW)","Random-Forest-(BoW)","SVM-(SVD)","Random-Forest-(SVD)","SVM-(W2V)","Random-Forest-(W2V)","My-Method"])
		writer.writerows(scores)


def main():

	args = make_args_parser()
	print_config(args)

	# Feeding the preprocessed_csv to avoid unnecessary overhead
	scores = []
	classification_out_dir = args.outputs + '/' + 'classification_out_dir/'

	# Experiment 1 : Support Vector Machines - BoW
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="BoW", kfold=True)
	scores.append(exp.run())

	# Experiment 2 : Random Forest - BoW
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="BoW", kfold=True)
	scores.append(exp.run())

	# Experiment 3 :  Support Vector Machines - SVD
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="SVD", kfold=True)
	scores.append(exp.run())

	# Experiment 4 :  Random Forest - SVD
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="SVD", kfold=True)
	scores.append(exp.run())
	
	# Experiment 5 :  Support Vector Machines - W2V
	exp = TextMining(args.datasets, args.outputs, classification="SVM", features="W2V", kfold=True)
	scores.append(exp.run())

	# Experiment 6 :  Random Forest - W2V
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="W2V", kfold=True)
	scores.append(exp.run())

	# Experiment 6 :  Random Forest - W2V
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="W2V", kfold=True)
	scores.append(exp.run())

	# Experiment 6 :  My Method
	exp = TextMining(args.datasets, args.outputs, classification="RF", features="W2V", kfold=True)
	scores.append(exp.run())

	PrintEvaluationFile(scores, classification_out_dir)


if __name__ == '__main__':
	main()


