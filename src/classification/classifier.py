import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import gensim

from preprocess.preprocess import *

class Classifier:

    def __init__(self, path, train_df, test_file, kfold, features="BoW"):

        test_df = pd.read_csv(test_file, sep='\t')

        self.le = preprocessing.LabelEncoder()
        self.le.fit(train_df['Category'])

        self.Y_train = self.le.transform(train_df['Category'])
        self.X_train = train_df['Content']

        self.X_test = test_df['Content']
        self.test_ids = test_df['Id']

        self.kfold = kfold

        self.features = "BoW" if features is None else features
        self.path = path
        self.tasks = []

    # self.classes = self.train_df.Category.unique()

    def populate_features(self):

        if self.features is "W2V":
            dim = 100
            self.X_train = Preprocessor().tokenize_articles(self.X_train)
            model = gensim.models.Word2Vec(self.X_train, size=dim)
            w2v = dict(zip(model.wv.index2word, model.wv.syn0))
            self.tasks.append(('vect', MeanEmbeddingVectorizer(w2v, dim)))
            # print(self.X_train)
            if not self.kfold:
                self.X_test = Preprocessor().tokenize_articles(self.X_test)
        elif self.features is "BoW":
            self.tasks.append(('vect', CountVectorizer(stop_words='english')))
            self.tasks.append(('tfidf', TfidfTransformer()))
        elif self.features is "SVD":
            svd = TruncatedSVD(n_components=50, random_state=42)
            self.tasks.append(('svd', svd))

        return self.tasks

    def predict(self, pipeline, classifier):

        pipeline.fit(self.X_train, self.Y_train)
        predicted = pipeline.predict(self.X_test)
        predlabels = self.le.inverse_transform(predicted)
        self.PrintPredictorFile(classifier, predlabels, self.test_ids, self.path)

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
            # print(metrics.classification_report(Ylabels, predlabels))

        self.PrintEvaluationFile(classifier, score_array, accuracy_array, self.path)

    def run(self):
        pass

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

    def PrintEvaluationFile(self, name, score_array, accuracy_array, path):

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
