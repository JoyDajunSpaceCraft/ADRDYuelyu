import pandas as pd
import json
from nltk.tokenize import sent_tokenize
from copy import deepcopy

original_question_tsv = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask3/question_answer.tsv"
title2index = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask3/title2index.json"
original_post = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

df = pd.read_csv(original_question_tsv, delimiter="\t")
# print(df["sentence"])

questions = df.loc[df["sentence"].str.contains("?", regex=False)]
title_id = list(questions["title_id"].values)
# print(questions["title_id"].values)
question2title = {}
for question, title_ids in zip(questions["sentence"].values, questions["title_id"].values):
    question2title[question] = title_ids  # assume that there is no duplicate question
    # print(question, title_id)


def genrate_json(title_id):
    final_output = []
    # here we get one_qa with format like this:
    # {
    #     "answers": [
    #         {
    #             "answer_start": 28,
    #             "text": "between 2006 and 2008"
    #         }
    #     ],
    #     "id": "00002",
    #     "is_impossible": False,
    #     "question": "When was the series published?"
    # },
    # one_qa = {}

    with open(title2index, "r", encoding="utf-8") as j_file:
        title_index = json.load(j_file)

    reverse_ = {}
    for key, value in title_index.items():
        reverse_[value] = key
    figure_out_title = []
    id2content = {}
    # print("title_id",title_id)
    for i in title_id:
        i = str(i)
        figure_out_title.append(title_index[i])

    # get all posts
    with open(original_post, "r", encoding="utf-8") as original_p:
        original_ = json.load(original_p)

    id = 0
    # first get the content by the title name and map content to title id
    content2titleid = {}

    for titles in figure_out_title:
        for i in original_:
            if titles in i["title"] and i["text"] != None:
                # qa_content["content"] = i["text"]
                content2titleid[i["text"]] = reverse_[titles]  # store in the format content : title id

    qa_content = {}
    qa_content = {}
    # one_qa = {}
    for question, title_id in question2title.items():
        one_qa = {}
        one_qa["question"] = question
        qa_content["qas"] = []
        if len(question.split(" ")) <= 9:
            # if the quetion sentence length more than 9
            # we think this sentence have enough information
            # get the keywords of the
            one_qa["is_impossible"] = False
            one_qa["id"] = str(id)
            id += 1
            answers = {}
            answers["answer_start"] = get_start_index(title_id, question, content2titleid, is_tier3=False)
            answers["text"] = " ".join(question.split(" ")[3:-1])
            one_qa["answers"] = [answers]
        else:
            one_qa["is_impossible"] = False
            one_qa["id"] = str(id)
            id += 1
            answer = {}
            answer["answer_start"], answer["text"] = get_start_index(title_id, question, content2titleid, is_tier3=True)
            # answer["text"] = " ".join(question.split(" ")[3:-1])
            one_qa["answers"] = [answer]
        key_ = list(content2titleid.keys())
        val_ = list(content2titleid.values())

        qa_content["content"] = key_[val_.index(str(title_id))]
        temp_qa = deepcopy(one_qa)
        qa_content["qas"].append(temp_qa)
        final_output.append(qa_content)
        qa_content = {}

    print(final_output)

    with open("train.json", "w") as wri:
        json.dump(final_output, wri)


def get_start_index(title_id, question, content2titleid, is_tier3):
    # from title_id get the fill sentnece
    key_ = list(content2titleid.keys())
    val_ = list(content2titleid.values())
    # print(val_)
    idx = val_.index(str(title_id))
    content = key_[idx]

    index = content.index(question)
    if is_tier3 == True:
        sentences = sent_tokenize(content)
        question_index = None
        for idx, i in enumerate(sentences):
            if question in i or i in question:
                question_index = idx
                break
        if index + len(question) < len(content):
            text = " ".join(sentences[question_index: question_index + 2])
        else:
            text = " ".join(sentences[question_index - 2: question_index])
        return index, text
    else:
        return index


# this function is to check the index of the answer_start in dev test train
def check_index():
    dev_path = "/ADRDtask3/onlyDailycare/dev_qa.json"
    test_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask3/test_qa.json"
    train_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask3/train_qa.json"
    with open(dev_path, "r") as dev:
        dev_json = json.load(dev)
    for item in dev_json:
        context = item["context"]
        answers = item["qas"][0]["answers"]
        for i in answers:
            index = context.index(i["text"])
            i["answer_start"] = index

    with open(test_path, "r") as test:
        test_json = json.load(test)
    for item in test_json:
        context = item["context"]
        answers = item["qas"][0]["answers"]
        for i in answers:
            try:
                index = context.index(i["text"])
                i["answer_start"] = index
            except Exception as e:
                print(context)
                print("item is", i["text"])

    with open(train_path, "r") as train:
        train_json = json.load(train)
    for item in train_json:
        context = item["context"]
        answers = item["qas"][0]["answers"]
        for i in answers:
            try:
                index = context.index(i["text"])
                i["answer_start"] = index
            except Exception as e:
                print(context)
                print("*" * 10, i["text"])

    # with open(dev_path,"w", encoding="utf-8") as dev_file:
    #     json.dump(dev_json, dev_file)

    # with open(test_path, "w", encoding="utf-8") as test_file:
    #     json.dump(test_json, test_file)

    with open(train_path, "w", encoding="utf-8") as train_file:
        json.dump(train_json, train_file)


def merge_question():
    pred_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask3/pred_json.json"
    all_post = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

    post_summary = {}
    with open(pred_path, "r") as pred_json:
        pred_ = json.load(pred_json)

    with open(all_post, "r") as all_posts:
        all_p = json.load(all_posts)

    title2context = {}
    for post in all_p:
        title2context[post["title"]] = post["text"].strip()

    for context, qa in pred_.items():
        question = qa["question"]
        answer = qa["answer"]

        if answer is None:
            print("the answer in {} is None".format(question))
            continue
        else:
            q_a = " ".join([question, answer])
        title = get_key_by_value(title2context, context)
        if title is None:
            print("The post {} have error inside, check the value is !".format(context))
            continue
        else:
            print("titles", title)
            post_summary[title] = q_a

    print("post_summary",post_summary)

    sentences = []
    titles = []
    for title, qas in post_summary.items():
        all_sents = sent_tokenize(qas)
        for i in all_sents:
            sentences.append(i)
            titles.append(title)


    df_csv = pd.DataFrame({"sentence":sentences,"title":titles,"label":[0]*len(sentences)})
    df_tsv = pd.DataFrame({"sentence":sentences,"label":[0]*len(sentences)})
    df_csv.to_csv("after_pred_sentences.csv",index=False)
    df_tsv.to_csv("after_pred_sentences.tsv",sep="\t",index=False)




def get_key_by_value(dicts, value):
    if value in dicts.values():

        for k, v in dicts.items():
            v = v.lower()
            value =value.lower()
            if v == value:
                print(v)
                print(value)
                return k
    else:
        return None


if __name__ == "__main__":
    genrate_json(title_id)
    # check_index()
    # merge_question()
