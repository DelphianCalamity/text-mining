from preprocess.preprocess import *
from classification.meanEmbeddingVectorizer import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from classification.classifier import Classifier

from sklearn.pipeline import Pipeline


class CustomClassifier(Classifier):

    def __init__(self, path, train_df, test_df, features):
        Classifier.__init__(self, path, train_df, test_df, features)

    def populate_features(self):
        tasks = Classifier.populate_features(self)

        # Add classifier task
        clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        clf2 = RandomForestClassifier(n_estimators=100, criterion='entropy')
        clf3 = LinearSVC()

        tasks.append(('clf', VotingClassifier(estimators = [('knn', clf1), ('rf', clf2), ('svm', clf3)], voting='soft', weights=[4, 2, 5])))
        self.pipeline = Pipeline(tasks)

    def run_kfold(self):
        self.populate_features()
        return self.k_fold_cv(self.pipeline)

    def run_predict(self):
        self.populate_features()
        return self.predict(self.pipeline, "RandomForests")
