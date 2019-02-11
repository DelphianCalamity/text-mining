from sklearn import preprocessing
import pandas as pd
import numpy as np
from word_cloud import generate_wordclouds
from duplicates_detection import detect_duplicates
import os.path

# Input - paths
csv_train_file = './datasets/train_set.csv'
csv_test_file = './datasets/test_set.csv'

# Output - paths
wordcloud_path = 'wordclouds/'
duplicates_path = 'duplicates/'

if not os.path.exists(wordcloud_path): os.makedirs(wordcloud_path)
if not os.path.exists(duplicates_path): os.makedirs(duplicates_path)

#Read Data
df = pd.read_csv(csv_train_file, sep='\t')

le = preprocessing.LabelEncoder()
le.fit(df['Category'])

# WordCloud creation
# generate_wordclouds(wordcloud_path, df, le.classes_)

X_train = df['Content']	
Y_train = le.transform(df['Category'])

# Duplicates Detection
# print(X_train)
detect_duplicates(X_train, 0.7, duplicates_path)

