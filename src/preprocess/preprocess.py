#from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
import re

"""
    Preprocessor class is the base class for preprocessing the input texts

    TODO: describe Preprocessor functionality
"""

class Preprocessor:

    def __init__(self, input_df, classes):
        self.df = input_df
        self.classes = classes
    
    # Joins all articles of a given category into a single string
    def _join_text(self, label):
        return ' '.join(self.df.loc[self.df['Category'] == label]['Content'].values)

    # Given a category, return its joined article content
    def text_per_category(self):
        return {category: self._join_text(category) for category in self.classes}

    def _stemSentence(self, sentence):

        lancaster = LancasterStemmer()

        token_words = word_tokenize(sentence)
        token_words
        stem_sentence = []
      
        rex=re.compile(r'\D+')
        
        for word in token_words:
            if rex.match(word):             # Removes numbers
                stem_sentence.append(lancaster.stem(word))
                stem_sentence.append(" ")
        
        return "".join(stem_sentence)

    def text_lemmatization(self, train_df):

        for index, row in train_df.iterrows():
            train_df.at[index, 'Content'] = self._stemSentence(row['Content'])
            # print(train_df.at[index, 'Content'])
        return train_df

    def save_to_csv(self, path):

            self.df.to_csv(path_or_buf=path, index=False, sep='\t')
