import pandas as pd
import numpy as np

from sklearn import preprocessing
from wordcloud import WordCloud
from preprocess.preprocess import *

class WordCloudGen:

    def __init__(self, path, csv_train_file, classes):
        self.path = path
        self.csv_train_file = train_file

        self.train_df = pd.read_csv(self.csv_train_file, sep='\t')
        self.preprocessor = Preprocessor(self.train_df, classes)

    # Builds WordClouds - one per class
    def generate_wordclouds(self):

        print("Generating wordclouds in " + self.path)
        content_per_cat = self.preprocessor.text_per_category()
        for label in self.preprocessor.classes:
            text = content_per_cat[label]
            wordcloud = WordCloud(max_words=1000,max_font_size=40, margin=10,
                        random_state=1, width=840, height=420).generate(text)
            wordcloud.to_file(self.path + label + '_wordcloud.png')
