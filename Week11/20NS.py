from __future__ import print_function  # using python 3 print

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

print("start!")
newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=0) # load "train" dataset

stopwords = open("stopwordlist.txt").read().split()
original = open("vocabulary.txt").read().split()          # open "vocabulary.txt"
                                                            # vocabulary.txt = about 60,000 words in 20newsgroups
vocabulary = []

for i in range(len(original)):
    if len(original[i])>2:
        vocabulary.append(original[i])

vectorizer = TfidfVectorizer(vocabulary=vocabulary, stop_words=stopwords)
#vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words=stopwords)         # the way of count words
raw_data = vectorizer.fit_transform(newsgroups_train.data)  # count words in each news
Y = newsgroups_train.target                                 # topic of each news

dataList = np.zeros(raw_data.shape)                         # make raw_data to ndarray
for i in range(raw_data.shape[0]):
    dataList[i] = raw_data[i].toarray()

dataSum = dataList.sum(axis=0)                              # sum => count words in all news
dataIndex = np.argsort(-dataSum)                            # np.argsort = sorting arg in ascending order, so make "minus" datasum
bagIndex = dataIndex[:2000]                                 # top 2000 words' index

bagVocab = np.array(vocabulary)[bagIndex]                   # top 2000 words

np.save('bagVocab_3+_newstop_tfidf.npy', bagVocab)                           # save 'bagVocab' as npy file


print("end!")
