from sklearn.ensemble import RandomForestClassifier
from classification.classifier import Classifier
from sklearn.pipeline import Pipeline


class RandomForests(Classifier):

    def __init__(self, path, train_file, test_file, kfold, features):
        Classifier.__init__(self, path, train_file, test_file, kfold, features)

    def run(self):

        tasks = self.populate_features()
        print(tasks)

        # n_estimators=100, max_depth=2,random_state=0
        clf = RandomForestClassifier()

        tasks.append(('clf', clf))

        pipeline = Pipeline(tasks)

        if not self.kfold : 
            self.predict(pipeline, "RandomForests")
        else:
            self.k_fold_cv(pipeline, "RandomForests")
