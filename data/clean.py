import pandas as pd
import preprocessor as prep
import re
from nltk.corpus import stopwords


# - Obtain the raw text data from the tweet status object returned by tweepy.
# - Removal of URLs from the tweet text
# - Switching the text to lower case
# - Removal of any punctuations
# - remove hastag# but dont remove word as often it states the contest such as #Oscars
#	or  #Romania to win #Eurovision 
 

dataset = "./data_with_text.csv"
dataframe = pd.read_csv(dataset, header = 0)


def processing(text,url,hashtag,letters,stopword):
	if url:
		#remove url and emoji
		prep.set_options(prep.OPT.URL, prep.OPT.EMOJI)
		text = prep.clean(str(text))

	if hashtag:
		prep.set_options(prep.OPT.HASHTAG)
		text = prep.clean(str(text))

	if letters:
		#letters only
		 text = re.sub("[^a-zA-Z0-9]", " ", text)

	if stopword:
		#remove stopwords
		words = text.lower().split()
		stopword_set = set(stopwords.words("english"))
		meaningful_words = [w for w in words if w not in stopword_set]
		text = " ".join(meaningful_words)

	return " ".join(text.lower().split())


if __name__ == '__main__':
	for i in range(len(dataframe)):
		dataframe.iloc[i,2] = processing(dataframe.iloc[i,2],True,False,True,False)
	dataframe.to_csv("preprocessed_data.csv", encoding='utf-8', index=False)
	
