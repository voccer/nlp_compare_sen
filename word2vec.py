import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models import Word2Vec

# path data
pathdata = './data/new.txt'

def read_data(path):
    traindata = []
    sents = open(pathdata, 'r').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata

if __name__ == '__main__':
    train_data = read_data(pathdata)
    model = Word2Vec(train_data, size=200, window=3, min_count=2, workers=4, sg=0)
    print(model)
    # summarize vocabulary

    model.save('model.bin')

    new_model = Word2Vec.load('model.bin')
    print(new_model)
    a = "tích_cực"
    print(new_model.wv.__getitem__(a))
    sim_words = model.wv.most_similar(a)
    print(sim_words)
