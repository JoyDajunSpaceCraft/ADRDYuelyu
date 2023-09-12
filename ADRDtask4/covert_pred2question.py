import pandas as pd
import numpy as np
import json
from copy import deepcopy
from nltk import sent_tokenize

# This function is make zero_shot_pred.txt file into the qustion_title.tsv
# the qustion_title file is ordered: title, sentences

def extract_pred():
    pred = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/pred_12000.txt"
    title_setence_hiw = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask5/all_post_title_sentence_hiw.tsv"
    df = pd.read_csv(title_setence_hiw, sep="\t")
    df = df[:12000] # here I just extract first 12000
    res_list = []

    with open(pred, "r") as predict:
        res = predict.readlines()
        # res = res[1:]

        for index, item in enumerate(res):
            res_list.append(item.strip().split(" "))

    question = {}

    for index, _, label in res_list:
        if label == "1":
            index = int(index)
            title_index = df.iloc[index].title
            title_index = str(title_index)
            if title_index not in question.keys():
                question[title_index] = [df.iloc[index].sentence]
            else:
                question[title_index].append(df.iloc[index].sentence)

    final_question = []
    for key, value in question.items():
        for j in value:
            final_question.append([key, j])
    final_question = np.array(final_question)
    write_in = pd.DataFrame(final_question)

    write_in.to_csv("question_title.csv", index=False, columns=["title","sentence"])

# this function is to generate the question answer pair
def convert_question():
    # title_post_class = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/200_title_post_class.json"
    title_post_class = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask5/all_post_title_post_class.json"

    # question_title = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask4/question_title.csv"
    question_title = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/question_title_12000.csv"


    df = pd.read_csv(question_title)
    questions = df.loc[df["sentence"].str.contains("?", regex=False)]
    # titles = list(questions["title"].values)
    question2title = {}

    # question2title[question: title]
    for question, title in zip(questions["sentence"].values, questions["title"].values):
        question2title[question] = title  # assume that there is no duplicate question
    final_output = []
    title2context = {}

    # get the dict {title: [context,label]}
    with open(title_post_class, "r") as json_file:
        title2context = json.load(json_file)

    qa_context = {}
    id = 0
    for question, title in question2title.items():
        one_qa = {}
        one_qa["question"] = question
        qa_context["qas"] = []

       # here we don't differ the long questions and shot questions
        one_qa["is_impossible"] = False
        one_qa["id"] = str(id)
        id += 1

        answers = []
        answer_text = get_answer(question, title, df)
        if len(answer_text)>1:
            for i in answer_text:
                subanswers = {}
                subanswers["text"] = i
                subanswers["answer_start"] = get_start_index(title, i, title2context)
                answers.append(subanswers)
        elif len(answer_text)==1:

            subanswers = {}
            subanswers["answer_start"] = get_start_index(title, answer_text[0], title2context)
            subanswers["text"] =answer_text[0]
            answers.append(subanswers)
        else:
            print("title {} have no answers".format(title))
        # answers.append(subanswers)


        one_qa["answers"] =answers

        qa_context["context"] = title2context[title][0]
        temp_qa = deepcopy(one_qa)
        qa_context["qas"].append(temp_qa)
        final_output.append(qa_context)
        qa_context = {}
    # print(final_output)
    with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/train.json", "w") as wri:
        json.dump(final_output, wri)


# tool function in convert_question
def get_start_index(title, answer, title2context):

    context = title2context[title][0]
    idx=None
    try:
        idx = context.index(answer)
    except Exception as e:
        print("the title {} can not get the index".format(title))
    return idx




def get_answer(question, title, df):
    answers = df.loc[df["title"].str.contains(title, regex=False) &  ~(df["sentence"].str.contains(question, regex=False))]
    # print("answers",list(answers.sentence))
    return list(answers.sentence)



def merge_pred2class():
    # because here I use the title as question so I just u
    predfile = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/pred_result_new/eletra_base_result.json"
    with open(predfile, "r") as preds:
        pred = json.load(preds)
    summary = {}
    for key, value in pred.items():
        que = value["question"]  # here we have 86 questions
        ans = value["answer"]

        summary[que] = " ".join([que, ans]) if ans else que
    title_post_class = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask5/all_post_title_post_class.json"
    with open(title_post_class, "r") as json_file:
        title2context = json.load(json_file)

    label_map = {"resource": 0, "daily care": 1, "psychosocial": 2, "treatment": 3, "legal": 4, "characteristics": 5, "care transition":6}

    label_list = []
    sentencen_list = []
    context_list = []
    for key, value in pred.items():
        que = value["question"]
        if que not in title2context.keys():continue
        context_label = title2context[que][1]
        append_label = label_map[context_label] if context_label in label_map.keys() else 0
        label_list.append(append_label)
        sentencen_list.append(summary[que])
        context_list.append(key)
    # cla = pd.DataFrame({"sentence":sentencen_list, "label":label_list,"context": context_list})
    cla = pd.DataFrame({"sentence":sentencen_list, "label":label_list})
    cla.to_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/eletra_base_train.tsv",sep="\t", index=False)



if __name__ == "__main__":
    merge_pred2class()
    # convert_question()
    # extract_pred()
    # convert_noquestion()