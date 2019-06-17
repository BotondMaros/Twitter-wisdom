import sys
import numpy as np 
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk.sentiment
from scipy.sparse import coo_matrix, hstack

def featureBuilder(path, origin=None):
    tweetFile = open(path, 'r', encoding='latin-1')
    tweetData = csv.reader(tweetFile)
    word = []
    c = 0
    y = []
    for data in tweetData:
        if c == 0:
            c+=1
            continue
        data[1] = data[1][2:-2].split('\', \'')
        tweet = re.sub("[^a-zA-Z0-9']", " ", data[2])
        tweet = nltk.sentiment.util.mark_negation(tweet.split())
        word.append(' '.join(tweet).lower())
        #word.append(tweet)
        verd = data[-1]
        if verd=='DY' or verd=='PY':
            verd = 1
        elif verd=='UC':
            verd = 0
        else:
            verd = -1
        y.append(verd)
    
    vectorizer = None
    if origin is None:
        vectorizer = CountVectorizer()
        feature = vectorizer.fit_transform(word)
    else:
        feature = origin.transform(word)
    c = 0
    newFt = []
    tweetFile = open(path, 'r', encoding='latin-1')
    tweetData = csv.reader(tweetFile)
    for data in tweetData:
        if c == 0:
            c+=1
            continue
        tweet = data[2]
        ft = []
        #ft.append(wordFeat(tweet))
        ft.append(EndsWithExclamation(tweet))
        ft.append(EndsWithQuestion(tweet))
        ft.append(EndsWithPeriod(tweet))
        ft.append(containsExclamation(tweet))
        ft.append(containsQuestion(tweet))
        tweet = re.sub("[^a-zA-Z0-9']", " ", tweet)
        ft.append(distanceToKwd(data))
        ft.append(negation(data))
        ft.append(negatedKwd(data))
        ft.append(kwdWin(data))
        newFt.append(ft)
    
    ft = coo_matrix(newFt)
    print(ft.sum(0))
    x = hstack([feature,ft])
    print(x.shape)
    return x,y, vectorizer
    
def kwdWin(tweet):
    tweetmod = nltk.sentiment.util.mark_negation(tweet[2].split())
    if len(tweet[1])>1 and "win" in tweetmod and tweet[1][1] in tweetmod:
        return 1
    else:
        return 0
    
#for data in array:
def wordFeat(tweet):
    feat = []
    start = -1
    end = -1
    win = -1
    strlis = tweet[2].split()
    if "win" in strlis:
        win = strlis.index("win")
        win = max(win-4, 0)
    if len(tweet[1]) > 2:
        if tweet[1][1] in strlis:
            start = strlis.index(tweet[1][1])
        if tweet[1][2] in strlis:
            end = strlis.index(tweet[1][2])
    start = min(start, end)
    end = max(start, end)
    if end-start != 0:
        if win == -1:
            return strlis[start:end+1]
        else:
            feat = strlis[start:end+1]
            end = min(len(strlis), win+8)
            for i in range(win, end):
                if strlis[i] in feat:
                    continue
                feat.append(strlis[i])
            return feat
    elif win == -1:
        return feat
    else:
        end = min(len(strlis), win+8)
        return strlis[win:end]
    
def threshold():
    return 1-0.64

def EndsWithExclamation(tweet):
        last = tweet[-1]
        if last == "!": 
            return 1
        else:
            return 0

def EndsWithQuestion(tweet):
        last = tweet[-1]
        if last == "?": 
            #print("haha")
            return 1
        else:
            return 0
                      
def EndsWithPeriod(tweet):
        last = tweet[-1]
        if last == ".": 
            return 1
        else:
            return 0

def containsQuestion(tweet):        
        if "?" in tweet:
            return 1
        return 0
                      
def containsExclamation(tweet):
        if "!" in tweet:
            return 1
        return 0                    

def distanceToKwd(tweet):
    strlis = tweet[2].lower().split()
    if len(tweet[1]) < 2:
        return 0
    if "win" in strlis and tweet[1][1] in strlis:
        return max(1,20-abs(strlis.index("win") - strlis.index(tweet[1][1])))
    else:
        return 0

def negation(tweet):
        tweetmod = nltk.sentiment.util.mark_negation(tweet[2].split())
        tweetmod = ' '.join(tweetmod).lower()
        if "_neg" in tweetmod:
            return 1
        else:
            return 0
                   
def negatedKwd(tweet):
        tweetmod = nltk.sentiment.util.mark_negation(tweet[2].split())
        if "win_NEG" in tweetmod:
            return 1
        elif len(tweet[1])>1 and tweet[1][1]+"_NEG" in tweetmod:
            print('yes')
            return 1
        else:
            return 0
                        
