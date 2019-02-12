import pandas as pd
import os

from word_cloud.word_cloud import *
from duplicates.duplicates_detection import *

class TextMining:

    def __init__(self, datasets, outputs, preprocess=False, wordclouds=False, 
            dupThreshold=None, classification=False, features=None):
        self.datasets = datasets
        self.outputs = outputs
        self.preprocess = preprocess
        self.wordclouds = wordclouds
        self.dupThreshold = dupThreshold
        self.classification = classification
        self.features = features

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
        print("extra data preprocessing")

    def generate_wordclouds(self):
        print("generate wordclouds per category of the given dataset")
        wcGen = WordCloudGen(self.wordcloud_out_dir, self.csv_train_file)
        wcGen.generate_wordclouds()

    def find_similar_docs(self):
        print("find similar documents")
        dupDet = DuplicateDetection(self.duplicates_out_dir, self.csv_train_file, self.dupThreshold)
        dupDet.detect_duplicates()

    def run_classifiers(self):
        print("run classifiers with the selected features: " + self.features)

    def run(self):
        if self.preprocess:
            self.preprocess()

        if self.wordclouds:
            self.generate_wordclouds()

        if self.dupThreshold:
            self.find_similar_docs()

        if self.classification:
            self.run_classifiers()


#    #Read Data
#    df = pd.read_csv(csv_train_file, sep='\t')
#
#    le = preprocessing.LabelEncoder()
#    le.fit(df['Category'])
#
#    # WordCloud creation
#    # generate_wordclouds(wordcloud_path, df, le.classes_)
#
#    X_train = df['Content']	
#    Y_train = le.transform(df['Category'])
#
#    # Duplicates Detection
#    # print(X_train)
#    detect_duplicates(X_train, 0.7, duplicates_path)


        # label_encoder = preprocessing.LabelEncoder()
        # le.fit(df['Category'])
        # Y_train = le.transform(df['Category'])
