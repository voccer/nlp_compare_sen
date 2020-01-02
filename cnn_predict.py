import pprint
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import datetime
import time
import pdb
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import random
from processingdata import process_sen

# 3h23p
vpath = "./data/vocab-vn.txt"
V = 50000


# import nltk
# nltk.download('averaged_perceptron_tagger')

def load_vocab(vocab, V):
    with open(vocab, 'r', encoding='utf-8') as f:
        word2id, id2word = {}, {}
        cnt = 0
        for line in f.readlines()[:V]:
            pieces = line.split()
            if len(pieces) != 2:
                exit(-1)
            word2id[pieces[0]] = cnt
            id2word[cnt] = pieces[0]
            cnt += 1
    return word2id, id2word


def load_data(fname, w2id):
    """
    :return: data
            list of tuples (s1, s2, score)
            where s1 and s2 are list of index of words in vocab
    """

    def get_indxs(sentence, w2id):
        MAX_LEN = 200
        res = []
        sp = sentence.split()
        for word in sp:
            if word in w2id:
                res.append(w2id[word])
            else:
                res.append(V)  # unk
        # pad/cut to MAX_LEN
        if len(res) > MAX_LEN:
            res = res[:MAX_LEN]
        else:
            res += [V + 1] * (MAX_LEN - len(res))
        return res

    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            sp = line.split('@')
            sp[0] = sp[0].strip()
            sp[1] = sp[1].strip()

            s1 = get_indxs(sp[0], w2id)
            s2 = get_indxs(sp[1], w2id)
            if len(sp) > 2:
                y = float(sp[2].strip())
            else:
                y = 0
            data.append((s1, s2, y, sp[0], sp[1]))
    return data


def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    if len(ss1) + len(ss2) == 0:
        pdb.set_trace()
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score


def extract_absolute_difference(s1, s2):
    """t \in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    s1, s2 = s1.split(), s2.split()
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = abs(len(s1) - len(s2)) / float(len(s1) + len(s2))
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return [t1, t2, t3, t4, t5]


def extract_mmr_t(s1, s2):
    shorter = 1
    if (len(s1) > len(s2)):  shorter = 2

    s1, s2 = s1.split(), s2.split()
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = (len(s1) + 0.001) / (len(s2) + 0.001)
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = (cnt1 + 0.001) / (cnt2 + 0.001)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = (cnt1 + 0.001) / (cnt2 + 0.001)

    if shorter == 2:
        t1 = 1 / (t1 + 0.001)
        t2 = 1 / (t2 + 0.001)
        t3 = 1 / (t3 + 0.001)
        t4 = 1 / (t4 + 0.001)
        t5 = 1 / (t5 + 0.001)

    return [t1, t2, t3, t4, t5]


def extract_baseline_features(s1, s2):
    res = []
    for i in range(len(s1)):
        st1, st2 = s1[i], s2[i]
        if st1 == ' ' and st2 == ' ':
            res.append([0] * 11)
            continue
        tmp = []
        tmp.append(extract_overlap_pen(st1, st2))
        tmp.extend(extract_absolute_difference(st1, st2))
        tmp.extend(extract_mmr_t(st1, st2))
        res.append(tmp)
    return np.array(res)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        V = 50002  # 10000 + unk + pad
        D = 128  # word embedding size
        Cin = 1  # input channel
        ks = [1, 2, 3, 4, 5, 6]  # kernel size
        Cout = 20
        dropout = 0.2

        self.embed = nn.Embedding(V, D)
        self.conv = nn.ModuleList([nn.Conv2d(Cin, Cout, (k, 2 * D)).double() for k in ks])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(ks) * Cout + 11, 1).float()

    def forward(self, s1, s2, baseline_features):
        # s1: batch_size x maxlen
        x1 = self.embed(s1).double()
        x2 = self.embed(s2).double()
        input = torch.cat([x1, x2], 2)  # batch_size x maxlen x 2D
        input = input.unsqueeze(1)  # N x 1 x maxlen x 2D
        out = [F.relu(conv(input).squeeze(3)) for conv in self.conv]  # [(N x Cout x maxlen)] * len(ks)
        out = [F.max_pool1d(z, z.size(2)).squeeze(2) for z in out]  # [(N x Cout)] * len(ks)
        out = torch.cat(out, 1)  # N x len(ks)*Cout
        out = self.dropout(out).float()
        out = torch.cat([out, baseline_features], 1)
        out = self.fc(out).float()
        return out


# load w2id
w2id, id2w = load_vocab(vpath, V)

# load model
model = CNN()
model.load_state_dict(torch.load('./model/cnn-vn-3.mdl'))

model.eval()


def compare(sen1, sen2):
    value = {'sen1': sen1, 'sen2': sen2}
    # print(value)
    f = open('./data/client_1', 'w')
    f.write("{}".format(sen1))
    f.close()

    f = open('./data/client_2', 'w')
    f.write("{}".format(sen2))
    f.close()

    process_sen()
    data_client = load_data('./data/new_client', w2id)
    res = []
    cnt = 0
    for item in data_client:
        s1, s2, score, s1s, s2s = np.array([item[0]]), np.array([item[1]]), np.array([item[2]]), item[3], item[4]
        baseline_features = extract_baseline_features([s1s], [s2s])
        baseline_features = Variable(torch.from_numpy(baseline_features)).float()
        s1 = Variable(torch.from_numpy(s1))
        s2 = Variable(torch.from_numpy(s2))
        output = model(s1, s2, baseline_features)
        res.append(output.data.cpu().numpy()[0][0])
    if res[0] < 0:
        res[0] = 0.0
    if res[0] > 5:
        res[0] = 5.0
    return res[0]