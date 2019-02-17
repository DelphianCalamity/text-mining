from preprocess.preprocess import *
from classification.meanEmbeddingVectorizer import *
from sklearn.svm import LinearSVC
from classification.classifier import Classifier

from sklearn.pipeline import Pipeline


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_df, features):
		Classifier.__init__(self, path, train_df, test_df, features)

	def populate_features(self):	
		tasks = Classifier.populate_features(self)
		# Add classifier task
		# TODO: Use a non-linear SVM and  experiment with kernels svm.SVC(kernel='linear', C=1)
		tasks.append(('clf', LinearSVC()))
		self.pipeline = Pipeline(tasks)

	def run_kfold(self):
		self.populate_features()
		return self.k_fold_cv(self.pipeline)

	def run_predict(self):
		self.populate_features()
		return self.predict(self.pipeline, "SupportVectorMachines")
