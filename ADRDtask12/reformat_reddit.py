import json
import pandas as pd
import nltk
from nltk import sent_tokenize
import numpy as np
import re
import os
import math
from datetime import datetime
import praw


def get_comment():
    # initialize with appropriate values
    client_id = "pV-DoxK3yzaD8eJTuMBRWQ"
    client_secret = "dQvLvNze_-hu6DRhIU_Nsu6Od12E3A"
    username = "yuj49"
    password = "Fiona@1999"
    user_agent = "yuj49"

    # creating an authorized reddit instance
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         username=username,
                         password=password,
                         user_agent=user_agent)
    dailycare_csv = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/dailycare_all_csv.csv"
    df = pd.read_csv(dailycare_csv)
    # print(df.columns)
    for idx, i in enumerate(df.comment.to_list()):
        if type(i) == float or i == " " or len(i.split(" ")) < 3:
            submission_id = str(df.iloc[idx].url)
            submission_id = submission_id.split("/comments/")[1].split("/")[0]
            submissions = reddit.submission(submission_id)
            comment_ids = submissions.comments._comments
            comments = ""
            for i in comment_ids:
                comment = reddit.comment(str(i))
                comments_ = comment.body
                comments_ = str.lower(comments_)
                comments_ = re.sub(r'http\S+', '', comments_)
                comments_ = re.sub(r'\n', '', comments_)
                comments = " " + comments_
            df.loc[idx, "comment"] = comments
        print(df.iloc[idx].comment)

    # df.to_csv("dailycare_all_csv", index=False)


path_new_ADRD = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/dailycare.csv"

df = pd.read_csv(path_new_ADRD)

sentences = df.selfbody.to_list()
reddit_m = {}
reddit_m["conv_test"] = []
reddit_m["conv_train"] = []
reddit_m["conv_valid"] = []
meta_ids = 0


def use_question_answer():
    path1 = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/Alzheimers.csv"
    path2 = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/Dementia.csv"
    alzheimer = pd.read_csv(path1)
    dementia = pd.read_csv(path2)

    post = alzheimer["selftext"].to_list()
    comment = alzheimer["comment"].to_list()
    for i in post:
        if i is not None and len(i) > 10:
            print(i)
            break


def gerate_conv(index, i):
    temp_conv = {}
    temp_conv["hashtag_lst"] = []
    temp_conv["meta_lst"] = []
    temp_conv["text_lst"] = []

    left = i.split(" ")
    leftlen = len(left)
    sent_length = leftlen // 20
    idx = 0
    while idx * 20 < leftlen:
        if idx + 1 == sent_length:
            break
        temp_conv["text_lst"].append(" ".join(left[idx * 20: (idx + 1) * 20]))
        idx += 1
    if idx * 20 - leftlen < 20:
        temp_conv["text_lst"].append(" ".join(left[idx * 20:leftlen]))
    sent_length = len(temp_conv["text_lst"])

    temp_conv["hashtag_lst"] = ["null"] * sent_length
    temp_conv["meta_lst"] = [{"id": str(ids + index)} for ids in range(sent_length)]
    return temp_conv


def generate_new():
    for index, i in enumerate(sentences):
        if index < 2000:
            reddit_m["conv_train"].append(gerate_conv(index, i))
        elif index < 2200 and index > 2000:
            reddit_m["conv_valid"].append(gerate_conv(index, i))
        else:
            reddit_m["conv_test"].append(gerate_conv(index, i))

    with open("Topic_Disc-master/data/twitter-conv/trec/new_reddit.json", "w") as wri:
        json.dump(reddit_m, wri)


def clean():
    csv_file_base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    base_list = ["Alzheimers", "AgingParents", "Dementia"]

    # get new in the 2020 - 2022
    posts = []
    comments = []
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, [os.path.join(csv_file_base_path, i + ".csv") for i in base_list]), ignore_index=True)
    post = df.selftext.values.tolist()
    comment = df.comment.values.tolist()

    for p, c in zip(post, comment):
        p = str(p)
        c = str(c)
        if p is None and c is None or type(p) == str or p == "":
            continue
        else:
            p = str.lower(p)
            c = str.lower(c)
            p = re.sub(r'http\S+', '', p)
            p = re.sub(r'\n', '', p)
            c = re.sub(r'http\S+', '', c)
            c = re.sub(r'\n', '', c)
            posts.append(p)
            comments.append(c)

    # get in 2010 - 2020
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"
    comment_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_post_comment.json"
    with open(base_path, "r") as f:
        all_json = json.load(f)

    with open(comment_path, "r") as f:
        comment_json = json.load(f)

    for value in all_json:
        p = value["text"]
        title = value["post_id"]

        if title in comment_json.keys():
            c = " ".join(comment_json[title]["comment"])
            c = str.lower(c)
            c = re.sub(r'http\S+', '', c)
            c = re.sub(r'\n', '', c)
        else:
            c = None

        p = str.lower(p)
        p = re.sub(r'http\S+', '', p)
        p = re.sub(r'\n', '', p)
        if p is None or p == "[removed]" or p == "":
            continue
        posts.append(p)
        comments.append(c)

    # only get the dailycare one
    dailycare = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/all_dailycare.csv"
    dailycare_ = pd.read_csv(dailycare)
    sent = dailycare_["sentence"].to_list()

    new_df = pd.DataFrame({"post": posts, "comment": comments})

    new_df.to_csv("all_post_comment.csv", index=False)


def match_dailycare():
    res_post = []
    res_comment = []
    dailycare = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/all_dailycare.csv"
    dailycare_ = pd.read_csv(dailycare)
    sent = dailycare_["sentence"].to_list()
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/all_post_comment.csv")
    posts = df.post.to_list()
    comments = df.comment.to_list()
    for idx, i in enumerate(posts):
        if type(i) == str and i != "nan" and i != "" and i != "[deleted]":
            temp_i = sent_tokenize(i)[:2]
            for j in sent:
                j = str.lower(j)
                j = re.sub(r'http\S+', '', j)
                j = re.sub(r'\n', '', j)
                temp_j = sent_tokenize(j)[:2]
                try:
                    if len(temp_j) > 1 and temp_i[0] == temp_j[0] and temp_i[1] == temp_j[1]:
                        res_post.append(i)
                        res_comment.append(comments[idx])
                        print(i)
                        break
                except Exception as e:
                    break
        else:
            continue
    new_df = pd.DataFrame({"post": res_post, "comment": res_comment})

    new_df.to_csv("dailycare_post_comment.csv", index=False)


def get_aftermay():
    path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/After_May"
    list_path = os.listdir(path)
    posts = []
    comments = []
    urls = []
    for i in list_path:
        json_file = os.path.join(path, i)
        with open(json_file, "r") as f:
            j_f = json.load(f)
            for j in j_f:
                if int(j["created_utc"]) > 1653072537:
                    if j["selftext"] not in ["", None, "[removed]", "[deleted]"]:
                        p = j["selftext"]
                        p = str.lower(p)
                        p = re.sub(r'http\S+', '', p)
                        p = re.sub(r'\n', '', p)
                        p = re.sub(r'\t', '', p)
                        posts.append(p)
                        comments.append(" ")
                        urls.append(j["full_link"])
                else:
                    continue
    pd.DataFrame({"submission": posts, "comments": comments, "url": urls}).to_csv("data_after_May.csv", index=False)


def get_at_may():
    csv_file_base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    base_list = ["Alzheimers", "AgingParents", "Dementia", "AlzheimersGroup"]

    # get new in the 2020 - 2022
    posts = []
    comments = []
    urls = []
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, [os.path.join(csv_file_base_path, i + ".csv") for i in base_list]), ignore_index=True)
    post = df.selftext.values.tolist()
    comment = df.comment.values.tolist()
    groups = df.subreddit.values.tolist()
    names = df.name.values.tolist()
    for idx, (p, c) in enumerate(zip(post, comment)):
        p = str(p)
        c = str(c)
        if p == "nan" and c == "nan":
            continue
        elif p == "nan" and c != "nan":
            posts.append(" ")
            c = str.lower(c)
            c = re.sub(r'http\S+', '', c)
            c = re.sub(r'\n', '', c)
            c = re.sub(r'\t', '', c)
            comments.append(c)
            urls.append(map_url(names[idx], groups[idx]))
        else:
            p = str.lower(p)
            c = str.lower(c)
            p = re.sub(r'http\S+', '', p)
            p = re.sub(r'\n', '', p)
            p = re.sub(r'\t', '', p)
            c = re.sub(r'http\S+', '', c)
            c = re.sub(r'\n', '', c)
            c = re.sub(r'\t', '', c)
            posts.append(p)
            comments.append(c)
            urls.append(map_url(names[idx], groups[idx]))

    pd.DataFrame({"submission": posts, "comments": comments, "url": urls}).to_csv("data_at_May.csv", index=False)


def map_url(name, group):
    url = "https://www.reddit.com/r/{}/comments/{}".format(group, name)
    return url


def count_empty_comment():
    csv_file_base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    base_list = ["Alzheimers", "AgingParents", "Dementia"]

    posts = []
    comments = []
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, [os.path.join(csv_file_base_path, i + ".csv") for i in base_list]), ignore_index=True)
    post = df.selftext.values.tolist()
    comment = df.comment.values.tolist()
    count = 0
    temp_comment = []
    for idx, i in enumerate(post):
        com = comment[idx]
        if type(com) != float:
            com = str.lower(com)
            com = re.sub(r'http\S+', '', com)
            com = re.sub(r'\n', '', com)
            com = re.sub(r'\t', '', com)
            if com != "" or com != None:
                count += 1
                temp_comment.append(com)
    print(count)
    print(idx)
    pd.DataFrame({"comment": temp_comment}).to_csv("temp_comment.csv", index=False)


def get_before_may():
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"
    comment_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_post_comment.json"
    with open(base_path, "r") as f:
        all_json = json.load(f)

    posts = []
    comments = []
    urls = []
    with open(comment_path, "r") as f:
        comment_json = json.load(f)

    for value in all_json:
        p = value["text"]
        title = value["post_id"]

        if title in comment_json.keys():
            c = " ".join(comment_json[title]["comment"])
            c = str.lower(c)
            c = re.sub(r'http\S+', '', c)
            c = re.sub(r'\n', '', c)
            c = re.sub(r'\t', '', c)
            c = re.sub(r'\t', '', c)
        else:
            c = None

        p = str.lower(p)
        p = re.sub(r'http\S+', '', p)
        p = re.sub(r'\n', '', p)
        p = re.sub(r'\t', '', p)
        p = re.sub(r'\t', '', p)
        if p is None or p == "[removed]" or p == "":
            continue
        posts.append(p)
        comments.append(c)
        urls.append(value["url"])

    pd.DataFrame({"submission": posts, "comments": comments, "url": urls}).to_csv("data_before_May.csv", index=False)


def merge():
    # df = pd.concat(
    #     map(pd.read_csv, ["data_before_May.csv","data_after_May.csv","data_at_May.csv"]), ignore_index=True)
    # df.to_csv("all.csv", index=False)

    df = pd.read_csv("all.csv")
    post = df.submission.values.tolist()
    comment = df.comment.values.tolist()
    url = df.url.values.tolist()
    sentence = []
    label = []
    urls = []
    comments = []
    posts = []

    for idx, (p, c) in enumerate(zip(post, comment)):

        if type(p) != str or p in ["[removed]", "[deleted]", None] or len(p.split(" ")) <= 3:
            p = " "

        if type(c) != str or c in ["[removed]", "[deleted]", None] or len(c.split(" ")) <= 3:
            c = " "

        if c == " " and p == " ":
            continue

        sentence.append(p + " " + c)
        label.append(0)
        posts.append(p)
        comments.append(c)
        urls.append(url[idx])
    print(len(urls))
    print(len(label))
    pd.DataFrame({"sentence": sentence, "label": label}).to_csv("test.csv", index=False)
    pd.DataFrame({"submission": posts, "comment": comments, "url": urls}).to_csv("all.csv", index=False)


def reformat():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/test.tsv"
    data = pd.read_csv(path, sep="\t")
    sentence = []
    label = []
    for idx, i in enumerate(data):
        if len(i) == 1:
            sentence.append("")
            label.append("0")
        if len(i) >= 2:
            sentence.append(" ".join(i[:-1]))
            label.append("0")
        else:
            sentence.append(i[0])
            label.append("0")
    pd.DataFrame({"sentence": sentence, "label": label}).to_csv("test.tsv", sep="\t", index=False)


def clean_example():
    examples = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/xieexample.tsv"
    df = pd.read_csv(examples, sep="\t")
    df.sentence = df.sentence.apply(clean_sentence)
    df.to_csv("train.tsv", sep="\t", index=False)


def clean_sentence(sentence):
    sentence = str.lower(sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub(r'\n', '', sentence)
    sentence = re.sub(r'\t', '', sentence)
    return sentence


def pred():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/pred.txt"
    label = []
    with open(path, "r") as f:
        for idx, i in enumerate(f.readlines()):
            s = i.split(" ")[-1].strip()
            label.append(s)
    pd.DataFrame({"label": label}).to_csv("pred.csv", index=False)


def get_post_comment_number():
    basic_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_inside_cat"
    bp = os.listdir(basic_path)
    bp.remove(".DS_Store")
    client_id = "pV-DoxK3yzaD8eJTuMBRWQ"
    client_secret = "dQvLvNze_-hu6DRhIU_Nsu6Od12E3A"
    username = "yuj49"
    password = "Fiona@1999"
    user_agent = "yuj49"
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         username=username,
                         password=password,
                         user_agent=user_agent)
    count_comment = {i: 0 for i in bp}
    for i in bp:
        path = os.path.join(basic_path, i)
        # count_comment = 0
        with open(path, "r") as f:
            # times_ = []
            j_f = json.load(f)
            print("postnumber {} : {}".format(i, len(j_f)))
            post_id = []
            for itme in j_f:
                print()
                count_comment[i] += int(itme["num_comments"])
    print("comment ", count_comment)

    # max_time = max(times_)
    # min_time = min(times_)
    #
    # dt_max_object = datetime.fromtimestamp(max_time)
    # dt_min_object = datetime.fromtimestamp(min_time)
    # print(i)
    # print("max time =", dt_max_object)
    # print("min time =", dt_min_object)


def merge_dailycare_dev():
    dailycare_dev = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/dailycare_dev.csv"
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/adrd_all"

    Post = []
    title = []
    label = []
    for i in os.listdir(base_path):
        if "non_adrd" not in i:
            df = pd.read_csv(os.path.join(base_path, i),lineterminator='\n')
            temp_submssion = df.submission.values.tolist()
            temp_comment = df.comment.values.tolist()
            temp_title = df.title.values.tolist()
            temp_Post = []
            cur_title = []
            for indexs, (s, c, t) in enumerate(zip(temp_submssion, temp_comment, temp_title)):
                if (type(s) == float or s == "[reomved]" or s == "[delete]" or len(s.split(" ")) < 5) :
                    continue
                else:
                    s = s.replace("\n"," ")
                    temp_Post.append(s)
                    cur_title.append(t)
                assert len(temp_Post) == len(cur_title)
            Post.extend(temp_Post)
            title.extend(cur_title)
            label.extend([0] *len(temp_Post))

    final_df = pd.DataFrame({"Post": Post, "title":title,"label":label})
    final_df.to_csv(dailycare_dev, index=False)


def markout_add():
    dailycare_dev = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/bert_classification/dailycare_dev.csv"
    df = pd.read_csv(dailycare_dev)
    small_train = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/bert_classification/dailycare_train_small.csv"
    train_df = pd.read_csv(small_train)

    posts = df.Post.values.tolist()
    titles = df.title.values.tolist()
    Post_new = []
    label = []
    for idx, i in enumerate(posts):
        if "insurance" in str.lower(i) or "finance" in str.lower(i) or "insurance" in str.lower(titles[idx]) or "finance" in str.lower(titles[idx]):
            Post_new.append(i)
            label.append(0)
    new_df = pd.DataFrame({"Post": Post_new, "label":label})
    train_df = pd.concat([train_df, new_df])
    train_df.to_csv(small_train, index=False)


if __name__ == "__main__":
    # get_before_may()
    # get_aftermay()
    # get_at_may()
    # merge()
    # get_comment()
    # get_post_comment_number()
    # merge_dailycare_dev()
    markout_add()