import pandas as pd
import numpy as np

from sklearn import preprocessing
from wordcloud import WordCloud

class WordCloudGen:

    def __init__(self, path, train_file):
        self.path = path
        self.csv_train_file = train_file

        self.train_df = pd.read_csv(self.csv_train_file, sep='\t')
        
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.train_df['Category'])

        self.classes = label_encoder.classes_

    # Builds WordClouds - one per class
    def generate_wordclouds(self):

        for label in self.classes : 
            text = self.train_df.loc[self.train_df['Category'] == label]['Content'].values
            wordcloud = WordCloud().generate((np.array2string(text)))
            wordcloud.to_file(self.path + label + '_wordcloud.png')
