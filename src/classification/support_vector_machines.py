from preprocess.preprocess import *
from classification.meanEmbeddingVectorizer import *
from sklearn.svm import LinearSVC
from classification.classifier import Classifier


from sklearn.pipeline import Pipeline


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_file, kfold, features):
		Classifier.__init__(self, path, train_df, test_file, kfold, features)

	def run(self):
		tasks = self.populate_features()

		# Add classifier task
		clf = LinearSVC() # TODO: Use a non-linear SVM and  experiment with kernels svm.SVC(kernel='linear', C=1)
		tasks.append(('clf', clf))

		return Pipeline(tasks)

	def run_kfold(self):\
		pipeline = self.run()
		return self.k_fold_cv(pipeline)

	def run_predict(self):
		return self.predict(pipeline, "SupportVectorMachines")
