from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS


import re
import logging

"""
    Preprocessor class is the base class for preprocessing the input texts

    TODO: describe Preprocessor functionality
"""

class Preprocessor:

    def __init__(self, input_df=None, classes=None, transformation="lemmatization"):
        self.df = input_df
        self.classes = classes
        self.transform = transformation

    # Joins all articles of a given category into a single string
    def _join_text(self, label):
        return ' '.join(self.df.loc[self.df['Category'] == label]['Content'].values)

    # Given a category, return its joined article content
    def text_per_category(self):
        return {category: self._join_text(category) for category in self.classes}

    # Removes stop words from each trainset entry
    def exclude_stop_words(self, train_df):
        # print(STOPWORDS)
        for index, row in train_df.iterrows():
            temp = row['Content'].lower()
            removed = remove_stopwords(temp)
            train_df.at[index, 'Content'] = "".join(removed)
        return train_df

    def stem_sentence(self, sentence):

        stemmer = LancasterStemmer()
        # remove numbers and special characters
        rex = re.compile(r'[a-z]')
        # tokenize and stem
        token_words = word_tokenize(sentence)
        stem_sentence = [stemmer.stem(word) for word in token_words if rex.match(word)]

        return " ".join(stem_sentence)


    def lem_sentence(self, sentence):

        lemmatizer = WordNetLemmatizer()
        # remove numbers and special characters
        rex = re.compile(r'[a-z]')
        # tokenize and lemmatize
        token_words = word_tokenize(sentence)
        lem_sentence = [lemmatizer.lemmatize(word, pos="v") for word in token_words if rex.match(word) and word != 'n\'t']

        return " ".join(lem_sentence)


    def text_transform(self, train_df):

        if self.transform == "lemmatization":
            transform = self.lem_sentence 
        elif self.transform == "stemming":
            transform = self.stem_sentence
        else:
            logging.error('Unknown text transformation "%s"', self.transform)
            raise Exception("Unknown text transformation " + self.transform)

        for index, row in train_df.iterrows():
            train_df.at[index, 'Content'] = transform(row['Content'])
        return train_df


    def tokenize_articles(self, X_train):
        tokenized = []
        for row in X_train:
            tokenized.append(word_tokenize(row))
        return tokenized

    def save_to_csv(self, df, path):
        df.to_csv(path_or_buf=path, index=False, sep='\t')
