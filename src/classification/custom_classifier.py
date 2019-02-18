from preprocess.preprocess import *
from classification.meanEmbeddingVectorizer import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from classification.classifier import Classifier

from sklearn.pipeline import Pipeline


class CustomClassifier(Classifier):

    def __init__(self, path, train_df, test_df, features):
        Classifier.__init__(self, path, train_df, test_df, features)

    def populate_features(self):
        tasks = Classifier.populate_features(self)

        # Add classifier tasks
        clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200, random_state=1)
        clf2 = RandomForestClassifier(n_estimators=100, criterion='entropy')
        clf3 = SVC(kernel='linear', probability=True)

        tasks.append(('clf', VotingClassifier(estimators = [('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft', weights=[4, 2, 5])))
        self.pipeline = Pipeline(tasks)

    def run_kfold(self):
        self.populate_features()
        return self.k_fold_cv(self.pipeline)

    def run_predict(self):
        self.populate_features()
        return self.predict(self.pipeline)
