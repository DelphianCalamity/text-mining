import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from classification.classifier import Classifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn import metrics


class RandomForests(Classifier):

	def __init__(self, path, train_file, test_file, kfold, features):
		Classifier.__init__(self, path, train_file, test_file, kfold, features)

	def run(self):
		pass