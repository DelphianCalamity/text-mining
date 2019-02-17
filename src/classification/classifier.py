import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from preprocess.preprocess import *
from classification.svd_variance import *
import gensim


class Classifier:

    def __init__(self, path, train_df, test_df, features):

        self.le = preprocessing.LabelEncoder()
        self.le.fit(train_df['Category'])

        self.Y_train = self.le.transform(train_df['Category'])
        self.X_train = train_df['Content']

        self.X_test = test_df['Content']
        self.test_ids = test_df['Id']

        self.features = features
        self.path = path
        self.tasks = []
    	# self.classes = self.train_df.Category.unique()
    
    def populate_features(self):

        if self.features == "W2V":
            dim = 100
            self.X_train = Preprocessor().tokenize_articles(self.X_train)
            model = gensim.models.Word2Vec(self.X_train, size=dim)
            w2v = dict(zip(model.wv.index2word, model.wv.syn0))
            self.tasks.append(('vect', MeanEmbeddingVectorizer(w2v, dim)))
            # print(self.X_train)
            if not self.kfold:
                self.X_test = Preprocessor().tokenize_articles(self.X_test)
        else:
            self.tasks.append(('vect', CountVectorizer(stop_words='english')))
            self.tasks.append(('tfidf', TfidfTransformer()))

        if self.features == "SVD":
            svd = TruncatedSVD(n_components=3500)
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
        self.PrintPredictorFile(classifier, predlabels, self.test_ids, self.path)
        return None

    def k_fold_cv(self, pipeline, classifier):
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

        avg_accuracy = np.mean(accuracy_array)
        precision_row = score_array[0]
        avg_precision = np.mean(precision_row, axis=0)
        recall_row = score_array[1]
        avg_recall = np.mean(recall_row, axis=0)

        print("Accuracy: " + avg_accuracy + "Precision: " + avg_precision + "Recall: " + avg_recall + "\n")
        return avg_accuracy, avg_precision, avg_recall


    def PrintPredictorFile(self, name, predicted_values, Ids, path):

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

