import os
import pandas as pd
import string
from pyvi import ViTokenizer
#modules to build the topic extracting models
from gensim import corpora, models
import gensim
import pyLDAvis.gensim
import pandas as pd
import matplotlib.pyplot as pl
# path data
pathdata = './data/test1'

def read_data(path):
    traindata = []
    sents = open(path, 'r').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata
if __name__ == '__main__':
    train_data = read_data(pathdata)
    #print(train_data)
    dictionary = corpora.Dictionary(train_data)
    dictionary.save("dicl.bin")
    print('{} different terms in the corpus'.format(len(dictionary)))
    #creating the bag of words object
    bow_corpus = [dictionary.doc2bow(text) for text in train_data]


    tfidf_model = models.TfidfModel(bow_corpus) # creating the tf-idf model
    tfidf_corpus = tfidf_model[bow_corpus]

    total_topics = 50
    lda_model_tfidf = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=total_topics,
                                  passes=1, random_state=47)
    lda_model_bow = models.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=total_topics,
                                passes=1, random_state=47)
    data = pyLDAvis.gensim.prepare(lda_model_bow, bow_corpus, dictionary)
    lda_model_bow.save('ldabow.bin')
    lda_model_tfidf.save('ldaifidf.bin')
    print("done")
    print(train_data[0])
    vec_bow_17 = dictionary.doc2bow(train_data[0])
    print(vec_bow_17)
    vec_bow_03 = dictionary.doc2bow(train_data[1])
    loadlda = models.LdaModel.load("ldabow.bin")
    vec_lda_topics_17 = loadlda[vec_bow_17]
    vec_lda_topics17 = lda_model_bow[vec_bow_17]
    vec_lda_topics_03 = lda_model_tfidf[vec_bow_03]
    print ('document 03 topics: ', vec_lda_topics_17)




