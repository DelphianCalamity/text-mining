import pandas as pd
import numpy as np
from sklearn import preprocessing


class Classifier:

	def __init__(self, path, train_file, test_file, kfold):
		
		train_df = pd.read_csv(train_file, sep='\t')
		test_df = pd.read_csv(test_file, sep='\t')

		self.le = preprocessing.LabelEncoder()
		self.le.fit(train_df['Category'])

		self.Y_train = self.le.transform(train_df['Category'])
		self.X_train = train_df['Content']
		
		self.X_test = test_df['Content']

		self.kfold = kfold

        # self.classes = self.train_df.Category.unique()


	def run(self):
		pass