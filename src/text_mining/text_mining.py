import pandas as pd
import logging
import os

from classification.support_vector_machines import *
from classification.random_forests import *

from word_cloud.word_cloud import *
from duplicates.duplicates_detection import *

class TextMining:

    def __init__(self, datasets, outputs, preprocess=False, wordclouds=False,
                 dup_threshold=None, classification=None, features=None, kfold=False):
        self.datasets = datasets
        self.outputs = outputs
        self.preprocess = preprocess
        self.wordclouds = wordclouds
        self.dupThreshold = dup_threshold
        self.classification = classification
        self.features = features
        self.kfold = kfold

        self.csv_train_file = datasets + '/' + 'train_set.csv'
        self.csv_test_file = datasets + '/' + 'test_set.csv'
        
        self.train_df = pd.read_csv(self.csv_train_file, sep='\t')
        self.classes = self.train_df.Category.unique()

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

    def preprocess_data(self):
        print("..data preprocessing")
        # self.train_df = Preprocessor().text_stemming(self.train_df)
        proccessed_csv_file =  self.datasets + '/' + 'proccessed_train_set.csv'
        
        self.train_df = Preprocessor().text_lemmatization(self.train_df)
        Preprocessor().save_to_csv(self.train_df, proccessed_csv_file)

    def generate_wordclouds(self):
        print("..generate wordclouds per category of the given dataset")
        wcGen = WordCloudGen(self.wordcloud_out_dir, self.csv_train_file, self.classes)
        wcGen.generate_wordclouds()

    def find_similar_docs(self):
        print("..find similar documents")
        dupDet = DuplicateDetection(self.duplicates_out_dir, self.train_df, self.dupThreshold, self.classes)
        dupDet.detect_duplicates()

    def run_classifiers(self):
        features = "Bow" if self.features is None else self.features
        print("..run " + str(self.classification) + " classifier with the selected features: " + str(features))

        if self.classification == 'SVM':
            clf = SupportVectorMachines
        elif self.classification == 'RF':
            clf = RandomForests
        # elif self.classification == '_KNN_':  # _KNN_ is a placeholder for out benchmark-beating classifier
        #     cf = None
        else:
            logging.error('Unknown classifier "%s"', self.classification)

        classifier = clf(self.classification_out_dir, self.train_df, self.csv_test_file, self.kfold, self.features)
        classifier.run()

    def run(self):
        if self.preprocess:
            self.preprocess_data()

        if self.wordclouds:
            self.generate_wordclouds()

        if self.dupThreshold:
            self.find_similar_docs()

        if self.classification:
            self.run_classifiers()
