from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

import re

"""
    Preprocessor class is the base class for preprocessing the input texts

    TODO: describe Preprocessor functionality
"""


class Preprocessor:

    def __init__(self, input_df=None, classes=None):
        self.df = input_df
        self.classes = classes

    # Joins all articles of a given category into a single string
    def _join_text(self, label):
        return ' '.join(self.df.loc[self.df['Category'] == label]['Content'].values)

    # Given a category, return its joined article content
    def text_per_category(self):
        return {category: self._join_text(category) for category in self.classes}

    def stem_sentence(self, sentence):

        lancaster = LancasterStemmer()

        token_words = word_tokenize(sentence)
        token_words
        stem_sentence = []

        # remove numbers and special characters
        rex = re.compile(r'[a-zA-Z]')
        for word in token_words:
            if rex.match(word):
                stem_sentence.append(lancaster.stem(word))
                stem_sentence.append(" ")

        return "".join(stem_sentence)

    @staticmethod
    def lem_sentence(sentence):

        wordnet_lemmatizer = WordNetLemmatizer()

        token_words = word_tokenize(sentence)
        token_words
        lem_sentence = []

        stop_words = set(stopwords.words('english')) 
        # remove numbers and special characters
        rex = re.compile(r'[a-zA-Z]')
        for word in token_words:

            if rex.match(word) and not word in stop_words:
                lem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
                lem_sentence.append(" ")

        return "".join(lem_sentence)

    def text_stemming(self, train_df):

        for index, row in train_df.iterrows():
            train_df.at[index, 'Content'] = self.stem_sentence(row['Content'])
            # print(train_df.at[index, 'Content'])
        return train_df

    def text_lemmatization(self, train_df):

        for index, row in train_df.iterrows():
            train_df.at[index, 'Content'] = self.lem_sentence(row['Content'])
            # print(train_df.at[index, 'Content'])
        return train_df

    @staticmethod
    def tokenize_articles(X_train):
        tokenized = []
        for row in X_train:
            tokenized.append(word_tokenize(row))
        return tokenized

    @staticmethod
    def save_to_csv(df, path):
        df.to_csv(path_or_buf=path, index=False, sep='\t')
