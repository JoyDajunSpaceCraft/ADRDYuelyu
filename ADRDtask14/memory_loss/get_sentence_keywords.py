from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import json
import os
from collections import Counter
import math


def remove_stopwords(sent):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sent)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    # for w in word_tokens:
    #     if w not in stop_words:
    #         filtered_sentence.append(w)
    sentence = " ".join(filtered_sentence)
    print(sentence)
    return sentence


def remove_comma(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    sents = tokenizer.tokenize(sent)
    return " ".join(sents)



def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)



def tf_idf_main():
    corpus = []
    train_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/train_memory_loss.csv")
    test_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/test_memory_loss.csv")
    valid_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/valid_memory_loss.csv")

    true_train = train_m.loc[train_m["label"]==1].Post_title.values.tolist()
    true_test = test_m.loc[test_m["label"]==1].Post_title.values.tolist()
    true_valid = valid_m.loc[valid_m["label"]==1].Post_title.values.tolist()
    corpus =true_train + true_test + true_valid

    words_list = list()
    for i in range(len(corpus)):
        words_list.append(corpus[i].split(' '))
    print(words_list)
    count_list = list()
    for i in range(len(words_list)):
        count = Counter(words_list[i])
        count_list.append(count)
    ti_idf = {}
    for i, count in enumerate(count_list):
        print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
        print("original file is ",corpus[i])

        scores = {word: tf_idf(word, count, count_list) for word in count}
        sorted_word = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        words_score = []
        for word, score in sorted_word:
            words_score.append([word, round(score, 5)])
            print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
        ti_idf[corpus[i]] = words_score

    with open("tfidf.json", "w") as jsonfile:
        json.dump(ti_idf, jsonfile)



if __name__ == "__main__":
    train_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_gpt/train_memory_loss.csv")
    test_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_gpt/test_memory_loss.csv")
    valid_m = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_gpt/valid_memory_loss.csv")
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc"
    name_list = ["train_memory_loss.csv","test_memory_loss.csv","valid_memory_loss.csv" ]

    for idx, dfs in enumerate([train_m, test_m, valid_m]):
        sents = dfs.Post.values.tolist()
        titles = dfs.title.values.tolist()
        corpus = []

        for i, j in zip(sents, titles):
            sents = " ".join([i, j])
            sents = remove_stopwords(sents)
            corpus.append(remove_comma(sents))

        dfs["Post_title"] = corpus
        dfs.to_csv(os.path.join(base_path, "{}".format(name_list[idx])), index=False)

    # tf_idf_main()