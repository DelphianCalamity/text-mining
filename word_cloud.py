import pandas as pd
import numpy as np
from wordcloud import WordCloud


# Builds WordClouds - one per class
def generate_wordclouds(path, df, classes):

	for label in classes : 
		text = df.loc[df['Category'] == label]['Content'].values
		wordcloud = WordCloud().generate((np.array2string(text)))
		wordcloud.to_file(path + label + '_wordcloud.png')