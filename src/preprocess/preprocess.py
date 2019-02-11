from nltk.stem.snowball import SnowballStemmer

class LightPreprocessor:

    def __init__(self, input_text_set):
        self.input_text_set = input_text_set
        self.train_df = pd.read_csv(self.input_text_set)
