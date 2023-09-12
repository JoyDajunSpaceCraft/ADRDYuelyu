import pandas as pd
import os
import time
from datetime import datetime
import json
import re
from nltk import sent_tokenize

s = "07/19/2022"
end_timestamp = time.mktime(datetime.strptime(s, "%m/%d/%Y").timetuple())
end_timestamp = int(end_timestamp)
# get all columns


column_dict = {"id": [], "created_utc": [], "full_link": [], "url": [], "selftext": [], "permalink": [],
               "title": [], "subreddit": [], "subreddit_id": [], "num_comments": []}
adrd_dir = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_inside_cat"

non_adrd_dir = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_other_cats"

id_list = []
create_time = []
full_link = []
url = []
selftext = []
permalink = []
title = []
subreddit = []
subreddit_id = []

adrd_list = os.listdir(non_adrd_dir)


# def clean_text(text):
#     s = str.lower(text)
#     s = re.sub(r'http\S+', '', s)
#     s = re.sub(r'\n', '', s)
#     return s
# this_column = list(column_dict.keys())
# this_column.remove("selftext")
# this_column.remove("created_utc")
# adrd_list.remove(".DS_Store")
# 
# 
# for i in adrd_list:
#     with open(os.path.join(non_adrd_dir, i), "r") as f:
#         all_json = json.load(f)
#         for item in all_json:
#             times = item["created_utc"]
#             times = int(times)
#             if times > end_timestamp:
#                 continue
#             if "selftext" not in item.keys():
#                 print("missing keys in ", item["id"])
#                 continue
#             column_dict["created_utc"].append(datetime.fromtimestamp(times))
#             column_dict["selftext"].append(clean_text(item["selftext"]))
#             for j in this_column: # all needed columns
#                 column_dict[j].append(item[j])
# 
# 
# 
# new_df = pd.DataFrame(column_dict)
# new_df.to_csv("no_adrd.csv", index=False)

# df = pd.read_csv("no_adrd_without_dementia.csv")
# selftexts = df.selftext.values.tolist()
# column_dicts = {"id":[], "created_utc":[], "full_link":[], "url":[], "selftext":[], "permalink":[],
#                "title":[], "subreddit":[], "subreddit_id":[],"num_comments":[]}
# id_lists = ["id", "created_utc", "full_link", "url","selftext", "permalink",
#                "title", "subreddit", "subreddit_id","num_comments"]
#
# for idx, i in enumerate(selftexts):
#     if type(i)!=float and ("dementia" in i or "alzheimer" in i):
#         for j in id_lists:
#             column_dicts[j].append(df.iloc[idx][j])
#     else:
#         continue
# new_df = pd.DataFrame(column_dicts)
# new_df.to_csv("no_adrd.csv", index=False)

# df = pd.read_csv("no_adrd.csv")
# name_list = {"CaregiverSupport":0,"AgingParents":0,"legaladvice":0,"Advice":0,"offmychest":0,"personalfinance":0,
#              "relationship_advice":0,"AskReddit":0,"GriefSupport":0,"confession":0,"AlzheimersSupport":0}
#
# subreddit = df.subreddit.values.tolist()
# for idx, i in enumerate(subreddit):
#     name_list[i]+=df.iloc[idx]["num_comments"]
# print(name_list)


def clean_non_adrd():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask13/no_adrd.csv"
    df = pd.read_csv(path)
    sents = df.selftext.values.tolist()
    columns = list(df.columns)
    res = {i:[]for i in columns}
    for idx, sent in enumerate(sents):
        if len(sent_tokenize(sent)) <= 12 and len(sent_tokenize(sent)) > 3:
         for i in columns:
            res[i].append(df.iloc[idx][i])
    pd.DataFrame(res).to_csv("clean_non_adrd.csv", index=False)


def match_with_title( original):
    origin = pd.read_csv(original)
    # p_title = preds.selftext.values.tolist()
    o_sents =  origin.selftext.values.tolist()
    columns = list(origin.columns)
    res = {i: [] for i in columns}

    for io, sent in enumerate(o_sents):
        if type(sent) ==float:
            continue
        sents = sent_tokenize(sent)
        len_sent = len(sents)
        if len_sent < 3 or len_sent > 12:
            continue
        last_30 = int(len_sent * 0.7)
        flag_question = False

        for lst in sents[last_30:]:
            if "?" in lst:
                flag_question = True
                break
        if flag_question:
            for i in columns:
                res[i].append(origin.iloc[io][i])
    pd.DataFrame(res).to_csv("clean_adrd.csv", index=False)


if __name__ == "__main__":
    # predict = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask13/new_new_result.csv"
    original= "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask13/adrd.csv"
    # clean_non_adrd()
    match_with_title(original)