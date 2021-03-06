from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

import pandas as pd

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

import re
import spacy
import logging


class Preprocessor:

    def __init__(self, input_df=None, classes=None, transformation="lemmatization"):
        self.df = input_df
        self.classes = classes
        self.transform = transformation
        self.stemmer = LancasterStemmer() if self.transform == "stemming" else None

    # Joins all articles of a given category into a single string
    def _join_text(self, label):
        return ' '.join(self.df.loc[self.df['Category'] == label]['Content'].values)

    # Given a category, return its joined article content
    def text_per_category(self):
        return {category: self._join_text(category) for category in self.classes}

    # Removes stop words from each trainset entry
    def exclude_stop_words(self, train_df, col='Content'):
        # print(STOPWORDS)
        for index, row in train_df.iterrows():
            temp = row[col].lower()
            removed = remove_stopwords(temp)
            train_df.at[index, col] = "".join(removed)
        return train_df

    def stem_sentence(self, sentence):

        # remove numbers and special characters
        rex = re.compile(r'[a-z]')
        # tokenize and stem
        stem_sentence = []
        token_words = word_tokenize(sentence)
        stem_sentence = [self.stemmer.stem(word) for word in token_words if rex.match(word)]

        return " ".join(stem_sentence)

    def lem_sentence(self, sentence):

        rex = re.compile(r'[a-z]')
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
        return " ".join([word.lemma_ for word in doc if rex.match(word.lemma_)])

    def text_transform(self, train_df, col='Content'):

        if self.transform == "lemmatization":
            transform = self.lem_sentence 
        elif self.transform == "stemming":
            transform = self.stem_sentence
        else:
            logging.error('Unknown text transformation "%s"', self.transform)
            raise Exception("Unknown text transformation " + self.transform)

        for index, row in train_df.iterrows():
                train_df.at[index, col] = transform(row[col])

        # Remove rows with "null" content
        train_df = train_df[pd.notnull(train_df[col])]
        
        return train_df

    def tokenize_articles(self, X_train):
        tokenized = []
        for row in X_train:
            tokenized.append(word_tokenize(row))
        return tokenized

    def save_to_csv(self, df, path):
        df.to_csv(path_or_buf=path, index=False, sep='\t')
