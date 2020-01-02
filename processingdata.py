from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd
from normalize import normalize_Text

# list stopsword

filename = './stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)
    return text2


def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def word_segment(sent):
    sent = tokenize(sent)
    return sent


def preprocess(line: str):
    line = normalize_Text(line)
    line = remove_stopword(line)
    line = word_segment(line)
    return line


def process_sen():
    f_1 = open('./data/client_1', 'r')
    f_2 = open('./data/client_2', 'r')
    f_a = open('./data/new_client', 'w')
    i = 0
    for line in f_1.readlines():
        if (line == None):
            continue
        line = normalize_Text(line)
        line = remove_stopword(line)
        line = word_segment(line)
        f_a.write(line + ' ')

    f_a.write('@')
    for line in f_2.readlines():
        if (line == None):
            continue
        line = normalize_Text(line)
        line = remove_stopword(line)
        line = word_segment(line)
        f_a.write(line + " ")

    f_1.close()
    f_2.close()
    f_a.close()