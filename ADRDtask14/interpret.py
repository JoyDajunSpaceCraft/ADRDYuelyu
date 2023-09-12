import nltk
import math
import string
import nltk.stem
from nltk.corpus import stopwords
from collections import Counter
# get attribution terms
import json
import pandas as pd



original_training = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/neg_pair_train_memory_loss.csv"
ot = pd.read_csv(original_training)

files = ot.Post.values.tolist()
labels = ot.label.values.tolist()

pos_corpus = []
neg_corpus = []

pos_idx = []
neg_idx = []

for index, (p, l) in enumerate(zip(files, labels)):
    if str(l) == "0":
        neg_corpus.append(p)
        neg_idx.append(index)
    else:
        pos_corpus.append(p)
        pos_idx.append(index)

punctuation_map = dict((ord(char), None) for char in string.punctuation)
s = nltk.stem.SnowballStemmer('english')



def stem_count(text):
    l_text = text.lower()
    without_punctuation = l_text.translate(punctuation_map)
    tokens = nltk.word_tokenize(without_punctuation)
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]

    count = Counter(without_stopwords)
    return count


# count tf-idf
def D_con(word, count_list):
    D_con = 0
    for count in count_list:
        if word in count:
            D_con += 1
    return D_con


def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    return math.log(len(count_list)) / (1 + D_con(word, count_list))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def generate_tfidf(new_negtxt, id_list):
    texts = new_negtxt
    count_list = []
    res = {}
    for text in texts:
        count_list.append(stem_count(text))  # fill clean up txt
    for idx, attr_tfidf in enumerate(id_list):
        # print('For document {}'.format(i+1))
        tf_idf = {}
        new_tfidf = {}
        for word in count_list[idx]:
            tf_idf[word] = tfidf(word, count_list[idx], count_list)

        res[int(id_list[idx])] = tf_idf
    return res


batchsize=1
import torch

max_padding_len = 768


def generate_epoch(res_tfidf, orignal_length):
    list_idx_original = [i for i in range(orignal_length)]

    match_batch_attr = []
    match_batch_tfidf = []

    for idx in range(orignal_length):
        if idx in lossed_list:
            match_batch_tfidf.append([0] * max_padding_len)
        else:
            # for (key_tfidf, word_dict_tfidf), (key_attr, word_list_attr)in zip(res_tfidf.items(), res_attribute.items()):
            word_dict_tfidf = res_tfidf[idx]
            tmp_tfidf = []
            tfidf_words = list(word_dict_tfidf.keys())  # ['found', 'manage',...]
            tfidf_scores = list(word_dict_tfidf.values())

            tmp_tfidf = normlize_score(tmp_tfidf)
            tmp_attr = normlize_score(tmp_attr)

            if len(tmp_tfidf) < max_padding_len:
                tmp_tfidf.extend([0] * (max_padding_len - len(tmp_tfidf)))



            match_batch_tfidf.append(tmp_tfidf)

    match_batch_tfidf = torch.FloatTensor(match_batch_tfidf)
    return match_batch_tfidf


def normlize_score(score_list):
    sums = 0
    for i in score_list:
        sums += i

    res = []
    for i in score_list:
        res.append(i / sums)
    return res

#
def get_word_sent(pos_res, neg_res):
    neg_ress = {}
    pos_ress = {}

    for k, value in pos_res.items():
        for word, score in value.items():
            pos_ress[word] = score
    for k, value in neg_res.items():
        for word, score in value.items():
            neg_ress[word] = score

    for k, value in neg_res.items():
        for word, score in value.items():
            if word in pos_ress.keys():
                neg_res[k][word] = [neg_res[k][word],pos_ress[word]]
            else:
                neg_res[k][word] = [neg_res[k][word],0]
    for k, value in pos_res.items():
        for word, score in value.items():
            if word in neg_ress.keys():
                pos_res[k][word] = [neg_ress[word], pos_res[k][word]]
            else:
                pos_res[k][word] = [0, pos_res[k][word]]

    for k, v in neg_res.items():
        pos_res[k] = v
    final_res = pos_res
    with open("final_res.json","w") as f:
        json.dump(final_res, f)
    return final_res


def get_info():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/final_res.json"
    with open(path, "r") as f:
        jsons = json.load(f)
    file1 = jsons["41"]
    file2 = jsons["43"]

    res = sorted(file1.items(), key=lambda item: item[1][1], reverse=True)
    for i in res:
        print(i[0])
        print(i[1][1])




if __name__ == "__main__":

    # neg_res = generate_tfidf(neg_corpus,neg_idx)
    # pos_res = generate_tfidf(pos_corpus,pos_idx)
    # final_res = get_word_sent(pos_res, neg_res)
    # res = {}
    # with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/final_res.json", "r") as f:
    #     final_res = json.load(f)
    # for k, value in final_res.items():
    #     sent_sum_neg = 0
    #     sent_sum_pos = 0
    #     for word, v in value.items():
    #         try:
    #             sent_sum_neg+=v[0]
    #             sent_sum_pos+=v[1]
    #         except Exception as e:
    #             print("s")
    #     res[k] = [sent_sum_neg, sent_sum_pos]
    # with open("tf_idf.json", "w") as f:
    #     json.dump(res,f)

    # with open("tf_idf.json","r") as f:
    #     jsons = json.load(f)
    #
    # jsons.load()
    get_info()
    # match_batch_tfidf, match_batch_attr = generate_epoch(pos_res, original_length)
