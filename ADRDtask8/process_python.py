import os

import pandas as pd
import json
from nltk import sent_tokenize
import numpy as np
pd.options.mode.chained_assignment = None

path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_post_comment.json"
# this function is to transfer the post and comment into the information question
# to xlsx file which ning can annotate
def trasfer_to_ning_csv():
    return_ = {}
    with open(path,"r") as json_:
        files = json.load(json_)
    title_text = []
    comments = []

    for key, i in files.items():
        title_text.append(" ".join(i["title_text"]))
        all_com = ""
        for idx, com in enumerate(i["comment"]):
            all_com+="comment {}: ".format(idx+1)+ com + "\n"
        comments.append(all_com)
    df = pd.DataFrame({"post":title_text,"comment":comments})

    df.to_csv("post_comment.csv", index=False)

def trasfer_post_comment():
    path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_post_comment.json"
    with open(path, "r") as f:
        files = json.load(f)

    final_title = {}
    final_label = {}
    titles = []
    contexts = []
    for key, i in files.items():
        title = i["title_text"][0]
        title_post = " ".join(i["title_text"])
        title_post+=" ".join(i["comment"])
        title_post = title_post.replace("\n","")
        title_post = title_post.replace("\t","")

        for i in sent_tokenize(title_post):
            titles.append(title)
            contexts.append(i)
    n = len(titles)
    final_title["title"] = titles
    final_title["context"] = contexts
    final_label["sentence"] = contexts
    final_label["label"] = [0]*n

    df_title = pd.DataFrame(final_title)
    df_label = pd.DataFrame(final_label)

    df_label.to_csv("text_label.tsv",index=False, sep="\t")
    df_title.to_csv("text_title.csv",index=False)


def convert2third():
    pred = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/pred.txt"
    res_list = []
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/text_title.csv")

    with open (pred, "r") as f:
        preds = f.readlines()
    for index, item in enumerate(preds):
        res_list.append(item.strip().split(" "))

    question = {}
    for index, _, label in res_list:
        if label == "1":
            index = int(index)
            title_index = df.iloc[index].title
            title_index = str(title_index)
            if title_index not in question.keys():
                question[title_index] = [df.iloc[index].context]
            else:
                question[title_index].append(df.iloc[index].context)
    final_question = []
    for key, value in question.items():
        for j in value:
            final_question.append([key, j])
    final_question = np.array(final_question)

    sentence_label = {}
    for i in final_question:
        if i[0] in sentence_label.keys():
            sentence_label[i[0]].append(str(i[1]))
        else:
            sentence_label[i[0]]= [str(i[1])]
    sentence = []

    for key, value in sentence_label.items():
        sentence.append(" ".join(value))

    n = len(sentence)
    label = [0]*n

    test_csv = pd.DataFrame({"sentence":sentence, "label":label})

    test_csv.to_csv("Clinical_test.tsv", index=False, sep="\t")



def process_dailycare():
    pred = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/pred_400.txt"
    text = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/test_400.tsv"
    predict = []
    with open(pred, "r") as f:
        for i in f.readlines():
            predict.append(i.strip().split(" "))

    df = pd.read_csv(text, sep="\t")

    for idx, _, label in predict:
        idx = int(idx)
        if label == "1":
            df.loc[idx,"label"] = "dailycare"
        else:
            df.loc[idx,"label"] = "None"
        print(df.iloc[idx].sentence)
    df.to_csv("dailycare_prediction.csv", index=False)






if __name__ == "__main__":
    process_dailycare()
    # convert2third()
    # trasfer_post_comment()







