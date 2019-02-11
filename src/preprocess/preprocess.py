#from nltk.stem.snowball import SnowballStemmer


"""
    LightPreprocessor class is the base class for preprocessing the input texts

    TODO: describe LightPreprocessor functionality
"""

class LightPreprocessor:

    def __init__(self, input_text_set=None, input_df=None, classes=None):
        self.input_text_set = input_text_set

        if self.input_text_set:
            self.df = pd.read_csv(self.input_text_set)
        else:
            self.df = input_df
            self.classes = classes
        self.content_per_cat = self._text_per_category()

    def _text_preprocess(self, text):
        return None

    # Joins all articles of a given category into a single string
    def _join_text(self, label):
        return ' '.join(self.df.loc[self.df['Category'] == label]['Content'].values)

    # Populates single string content per category
    def _text_per_category(self):
        return {category: self._join_text(category) for category in self.classes}

    # Given a category, return its joined article content
    def preprocessed_text_per_cat(self, category):
        return self.content_per_cat[category]

