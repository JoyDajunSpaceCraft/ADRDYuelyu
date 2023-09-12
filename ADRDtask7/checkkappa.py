import pandas as pd
from nltk import sent_tokenize
import math
import json

def write_question_answer():

    # with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask7/yuelyu_ning.csv","r")as f:
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask7/yuelyu_ning_second_round.csv")
    yuelyu_q = list(df["yuelyu question"])
    yuelyu_a = list(df["yuelyu answer"])

    # yuelyu_ag = list(df["yuelyu answer"])
    ning_a = list(df["ning answer"])
    ning_q = list(df["ning question"])

    # ning_ag = list(df["Ning answer"])
    # yuelyu_question  = list(df["yuelyu question"])
    truth = 0

    # generate aggreement in questions
    for i in range(len(ning_q)):
        isAdd_question = False
        isAdd_answer = False

        if type(ning_q[i]) is str:
            if (ning_q[i] is None or ning_q[i] == "N/a" or isinstance(ning_q[i], float) or ning_q[i]=="n/a") or \
                    (yuelyu_q[i] is None or yuelyu_q[i] == "N/a" or isinstance(yuelyu_q[i], float) or yuelyu_q[i]=="n/a"):
                df.loc[i, "Agreement question"] = 1
                continue
            if ning_q[i]!=None and yuelyu_q[i]!=None:
                yue_sent_q = sent_tokenize(yuelyu_q[i])
                ning_sent_q = sent_tokenize(ning_q[i])

                for m in yue_sent_q:
                    for n in ning_sent_q:
                        if m in n or n in m:
                            df.loc[i, "Agreement question"] = 1

                            truth += 1
                            isAdd_question = True
                            break
                    if isAdd_question == True:
                        break
            if isAdd_question is False:
                print("index is", i + 1)
                print("ning is: ", ning_q[i])
                print("yuelyu is: ", yuelyu_q[i])

                df.loc[i, "Agreement question"] = 0



        if type(ning_a[i]) is str:
            yue_sent_a = sent_tokenize(yuelyu_a[i])
            ning_sent_a = sent_tokenize(ning_a[i])
            for m in yue_sent_a:
                for n in ning_sent_a:
                    if m in n or n in m:
                        df.loc[i, "Agreement answer"] = 1

                        truth += 1
                        isAdd_answer = True
                        break

                if isAdd_answer == True:
                    break

            if isAdd_answer is False:
                # print("index is", i + 1)
                # print("ning is: ", ning_a[i])
                # print("yuelyu is: ", yuelyu_a[i])
                df.loc[i, "Agreement answer"] = 0

        # if isAdd is False and isinstance(ning_q[i], str):
        #     print("index {} ".format(i+2))
        #     print("yuelyu: {} ".format( yuelyu[i]))
        #     print("ning: {} ".format( ning[i]))

    # print("the current df is", df)
    # df.to_csv("disaggrement_second_round_2.csv", index=False)
    # print(truth)
    # print(truth/ 87)
    # 0.27586206896551724 the true value of both
    # true_ning = 0
    # true_yuelyu = 0
    # for i in range(len(ning)):
    #     if ning[i] == None:
    #         pass
    #     if isinstance(ning[i], float) and math.isnan(ning[i]):
    #         pass
    #     else:
    #         true_ning+=1
    #     if  yuelyu[i] == None:
    #         pass
    #     else:true_yuelyu+=1
    #
    #
    #
    # false_ning = 87 - true_ning
    #
    # false_yuelyu = 87- true_yuelyu
    #
    # print(true_ning) # 45 = 0.517
    # print(false_ning) # 42
    # print(true_yuelyu)
    # print(false_yuelyu)
    #

def count_kappa():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask7/disaggrement_second_round_kappa.csv"
    df = pd.read_csv(path)

    agree_question = list(df["Agreement question"])
    agree_answer = list(df["Agreement answer"])
    ques_count = 0
    ans_count = 0
    for i in range(len(agree_answer)):
        if agree_question[i] == 1.0:
            ques_count+=1
        if agree_answer[i] == 1.0:
            ans_count+=1
    print(ques_count/len(agree_answer))
    print(ans_count/len(agree_answer))
import numpy as np

# 没有对输入的合法性进行校验
# 使用时需要注意
def kappa(confusion_matrix):
    """计算kappa值系数"""
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)


def accuarcy():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask7/disaggrement_second_round_acc.csv"
    df = pd.read_csv(path)
    question_annotated = df["ning question"]
    model_question = df["question"]
    roBerta = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/pred_result_new/xlnet.json"
    with open (roBerta, "r") as f:
        json_ = json.load(f)
    question_roberta = {}
    for i in json_.items():
        item = i[1]
        question_roberta[item["question"]] = item["answer"]

    answer_annotated = df["ning answer"]
    model_answer = df["background"]
    answer_true = 0
    question_true = 0
    for i in range(len(model_answer)):
        isAdd_answer = False
        isAdd_question = False
        # question
        try:
            if type(question_annotated[i]) is str and(question_annotated[i] != None and model_question[i]!=None):
                question_an = sent_tokenize(question_annotated[i])
                model_qu = sent_tokenize(model_question[i])
                for m in question_an:
                    for n in model_qu:
                        if m in n or n in m:
                            # df.loc[i, "Agreement answer"] = 1
                            question_true += 1
                            isAdd_question = True
                            break
                    if isAdd_question == True:
                        break
        except Exception as e:
            print(e)
            # continue

        # answer
        try:
            answer_an = sent_tokenize(answer_annotated[i])
            model_ans = sent_tokenize(question_roberta[model_question[i]])
            for m in answer_an:
                for n in model_ans:
                    if m in n or n in m:
                        # df.loc[i, "Agreement answer"] = 1
                        answer_true += 1
                        isAdd_answer = True
                        break
                if isAdd_answer == True:
                    break
        except Exception as e:
            print(e)
            # continue

        if (type(answer_annotated[i]) is float or answer_annotated[i] is None) and (type(model_answer[i]) is float or model_answer[i] is None):
            answer_true+=1
    print(question_true)
    print(answer_true)

    print(question_true / 87)
    print(answer_true / 87)

if __name__ == "__main__":
    # count_kappa()
    # write_question_answer()
    accuarcy()