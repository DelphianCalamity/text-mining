from preprocess.preprocess import *
from sklearn.ensemble import RandomForestClassifier
from classification.classifier import Classifier

from sklearn.pipeline import Pipeline


class RandomForests(Classifier):

	def __init__(self, path, train_df, test_df, features):
		Classifier.__init__(self, path, train_df, test_df, features)

	def populate_features(self):    
		tasks = Classifier.populate_features(self)
		tasks.append(('clf', RandomForestClassifier(n_estimators = 100, criterion = 'entropy')))
		self.pipeline = Pipeline(tasks)

	def run_kfold(self):
		self.populate_features()
		return self.k_fold_cv(self.pipeline)

	def run_predict(self):
		self.populate_features()
		return self.predict(self.pipeline)
