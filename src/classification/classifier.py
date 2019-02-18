import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from classification.meanEmbeddingVectorizer import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from preprocess.preprocess import *
from classification.svd_variance import *
# from sklearn import metrics
import gensim


class Classifier:

	def __init__(self, path, train_df, test_df, features):

		self.le = preprocessing.LabelEncoder()
		self.le.fit(train_df['Category'])

		self.Y_train = self.le.transform(train_df['Category'])
		self.X_train = train_df['Content']

		# print(self.X_train)
		self.X_test = test_df['Content'] if not test_df is None else None
		self.test_ids = test_df['Id'] if not test_df is None else None

		self.features = features
		self.path = path
		self.tasks = []
		# self.classes = self.train_df.Category.unique()
	
	def populate_features(self):

		if self.features == "W2V":
			dim = 100
			X_train = Preprocessor().tokenize_articles(self.X_train)
			model = gensim.models.Word2Vec(X_train, size=dim)
			w2v = dict(zip(model.wv.index2word, model.wv.syn0))
			self.tasks.append(('vect', MeanEmbeddingVectorizer(w2v, dim)))

		else:
			self.tasks.append(('vect', CountVectorizer(stop_words='english')))
			self.tasks.append(('tfidf', TfidfTransformer()))

		if self.features == "SVD":
			svd = TruncatedSVD(n_components=5000)
			self.tasks.append(('svd', svd))
			self.tasks.append(('print_svd_variance', SvdVariancePrinter(svd)))

		return self.tasks

	def run_kfold(self):
		pass

	def run_predict(self):
		pass

	def predict(self, pipeline, classifier):

		pipeline.fit(self.X_train, self.Y_train)
		predicted = pipeline.predict(self.X_test)
		predlabels = self.le.inverse_transform(predicted)
		self.print_predictor_file(classifier, predlabels, self.test_ids, self.path)
		return None

	def k_fold_cv(self, pipeline):
		score_array = []
		accuracy_array = []
		kf = KFold(n_splits=10)

		for train_index, test_index in kf.split(self.X_train):
			pipeline.fit(self.X_train[train_index], self.Y_train[train_index])
			predicted = pipeline.predict(self.X_train[test_index])

			Ylabels = self.le.inverse_transform(self.Y_train[test_index])
			predlabels = self.le.inverse_transform(predicted)
			score_array.append(precision_recall_fscore_support(Ylabels, predlabels, average=None))
			accuracy_array.append(accuracy_score(Ylabels, predlabels))
			# print(metrics.classification_report(Ylabels, predlabels))
		
		avg_accuracy = round(np.mean(accuracy_array),4)
		avg_scores = np.mean(np.mean(score_array, axis=0), axis=1)
		avg_precision = round(avg_scores[0],4)
		avg_recall = round(avg_scores[1],4)
		# print(score_array)
		# print(avg_scores)
		
		print("Accuracy: " + str(avg_accuracy) + "\nPrecision: " + str(avg_precision) + "\nRecall: " + str(avg_recall) + "\n")
		return avg_accuracy, avg_precision, avg_recall

	def print_predictor_file(self, name, predicted_values, Ids, path):

		with open(path + name + '_testSet_categories.csv', 'w') as f:
			sep = '\t'
			f.write('Test_Document_ID')
			f.write(sep)
			f.write('Predicted Category')
			f.write('\n')

			for Id, predicted_value in zip(Ids, predicted_values):
				f.write(str(Id))
				f.write(sep)
				f.write(predicted_value)
				f.write('\n')
