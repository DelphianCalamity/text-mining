import pandas as pd
import numpy as np
from sklearn import preprocessing


class Classifier:

	def __init__(self, path, train_df, test_file, kfold, features):
		
		test_df = pd.read_csv(test_file, sep='\t')

		self.le = preprocessing.LabelEncoder()
		self.le.fit(train_df['Category'])

		self.Y_train = self.le.transform(train_df['Category'])
		self.X_train = train_df['Content']
		
		self.X_test = test_df['Content']
		self.test_ids = test_df['Id']
		
		self.kfold = kfold
		self.features = features
		self.path = path

        # self.classes = self.train_df.Category.unique()


	def run(self):
		pass


	def PrintPredictorFile(name, predicted_values, Ids, path):		
		
		with open(path + name + '_testSet_categories.csv', 'w') as f:
			sep = '\t'
			f.write('Test_Document_ID')
			f.write (sep)
			f.write( 'Predicted Category' )
			f.write('\n')

			for Id, predicted_value in zip(Ids, predicted_values):
				f.write( str(Id) )
				f.write( sep )
				f.write( predicted_value )
				f.write('\n')


	def PrintEvaluationFile(self, name, scores, accuracies, path):
		
		with open(path + name + 'EvaluationMetric_10fold.csv', 'w') as f:
			sep = '\t'

			avg_accuracy = np.mean(accuracy_array)
			f.write('Average Accuracy')
			f.write(sep)
			f.write(str(round(avg_accuracy, 3)))
			f.write('\n')
			avg_score = np.mean(score_array, axis=0)

			print(avg_score)
			print(avg_accuracy)

			# Precision
			precision_row = score_array[0]
			avg_precision = np.mean(precision_row, axis=0)
			f.write('Average Precision')
			f.write(sep)
			f.write(str(round(avg_precision, 3)))
			f.write('\n')

			# Recall
			recall_row = score_array[1]
			avg_recall = np.mean(recall_row, axis=0)
			f.write('Average Precision')
			f.write(sep)
			f.write(str(round(avg_recall, 3)))
			f.write('\n')