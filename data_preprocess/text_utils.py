# -*- coding:utf-8 -*-
from nltk.tokenize import word_tokenize
import jieba

def TokenizeText(text, stopwords):
    words = list(jieba.cut(text))
    words = [word for word in words if word not in stopwords]
    return words
    # return word_tokenize(text.lower())

def ParseNgramsOpts(opts):
    ngrams = [int(g) for g in opts.split(',')]
    ngrams = [g for g in ngrams if 1 < g < 7]
    return ngrams


def GenerateNgrams(words, ngrams):
    nglist = []
    for ng in ngrams:
        for word in words:
            nglist.extend([word[n:n+ng] for n in range(len(word)-ng+1)])
    return nglist
