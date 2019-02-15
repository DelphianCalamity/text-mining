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


	def PrintEvaluationFile(self, accuracy_values, path):
		
		with open(path + 'EvaluationMetric_10fold.csv', 'w') as f:
			sep = '\t'
			f.write('Accuracy')
			for accuracy_value in zip(*accuracy_values):
				f.write(sep)
				for x in accuracy_value:
					f.write(str(round(x,3)))
					f.write(sep)
				f.write('\n')


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