import pandas as pd
import logging
import os

from classification.support_vector_machines import *
from classification.random_forests import *

from word_cloud.word_cloud import *
from duplicates.duplicates_detection import *

class TextMining:

    def __init__(self, datasets, outputs, preprocess=False, wordclouds=False, 
            dupThreshold=None, classification=None, features=None, kfold=False):
        self.datasets = datasets
        self.outputs = outputs
        self.preprocess = preprocess
        self.wordclouds = wordclouds
        self.dupThreshold = dupThreshold
        self.classification = classification
        self.features = features
        self.kfold = kfold

        self.csv_train_file = datasets + '/' + 'train_set.csv'
        self.csv_test_file = datasets + '/' + 'test_set.csv'
        
        # define output directory names
        self.wordcloud_out_dir = outputs + '/' + 'wordcloud_out_dir/' if self.wordclouds else None
        self.duplicates_out_dir = outputs + '/' + 'duplicates_out_dir/' if self.dupThreshold else None
        self.classification_out_dir = outputs + '/' + 'classification_out_dir/' if self.classification else None

        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)

        if self.wordcloud_out_dir:
            if not os.path.exists(self.wordcloud_out_dir):
                os.makedirs(self.wordcloud_out_dir)

        if self.duplicates_out_dir:
            if not os.path.exists(self.duplicates_out_dir):
                os.makedirs(self.duplicates_out_dir)
       
        if self.classification_out_dir:
            if not os.path.exists(self.classification_out_dir):
                os.makedirs(self.classification_out_dir)

    def preprocess(self):
        print("..extra data preprocessing")

    def generate_wordclouds(self):
        print("..generate wordclouds per category of the given dataset")
        wcGen = WordCloudGen(self.wordcloud_out_dir, self.csv_train_file)
        wcGen.generate_wordclouds()

    def find_similar_docs(self):
        print("..find similar documents")
        dupDet = DuplicateDetection(self.duplicates_out_dir, self.csv_train_file, self.dupThreshold)
        dupDet.detect_duplicates()

    def run_classifiers(self):
        print("..run classifiers with the selected features: " + str(self.features))

        if self.classification == 'SVM':
            clf = SupportVectorMachines
        elif self.classification == 'RF':
            clf = RandomForests
        # elif self.classification == '_KNN_':  # _KNN_ is a placeholder for out benchmark-beating classifier
        #     cf = None
        else:
            logging.error('Unknown classifier "%s"', self.classification)

        classifier = clf(self.classification_out_dir, self.csv_train_file, self.csv_test_file, self.kfold)
        classifier.run()


    def run(self):
        if self.preprocess:
            self.preprocess()

        if self.wordclouds:
            self.generate_wordclouds()

        if self.dupThreshold:
            self.find_similar_docs()

        if self.classification:
            self.run_classifiers()