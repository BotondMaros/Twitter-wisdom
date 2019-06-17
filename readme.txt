Implementation of the paper: "i have a feeling trump will win..................": Forecasting Winners and Losers from User Predictions on Twitter
https://arxiv.org/abs/1707.07212


Necessary packages sklearn, tensorflow, bert_text pandas, nltk and scipy

In the data folder you can find 
1. scrape_twitter.py which was used to collect the data and create dataframes
2. clean.py to preprocess the tweet's text

In the reproduce folder you can find
1. runClassifier.py, with pretrained model file train.save and Vocab.save, which was provided by Swamy to yield his training result.
2. mainReproduce.py, which runs our recreated implementation on the proposed logistic regression model. Uses featureExtract.py.
3. featureExtract.py to construct the proposed feature set by Swamy et al.

Assuming that the data is prepared in train.csv and test.csv
1. To run bert_model, just type python bert_model.py
2. To run the nb-svm, type python .\nb-svm.py --train train.csv --test test.csv --ngrams 1,2
