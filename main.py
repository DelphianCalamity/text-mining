from sklearn import preprocessing
import pandas as pd
import numpy as np
from word_cloud import generate_wordclouds
import os.path

# Input - paths
csv_train_file = './datasets/train_set.csv'
csv_test_file = './datasets/test_set.csv'

# Output - paths
wordcloud_path = 'wordclouds/'

if not os.path.exists(wordcloud_path): os.makedirs(wordcloud_path)

#Read Data
df = pd.read_csv(csv_train_file, sep='\t')

le = preprocessing.LabelEncoder()
le.fit(df['Category'])


generate_wordclouds(wordcloud_path, df, le.classes_)

# X_train = df['Content']	
# Y_train = le.transform(df['Category'])
