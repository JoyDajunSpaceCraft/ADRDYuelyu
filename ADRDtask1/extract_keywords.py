import os
import json
import pandas as pd
import csv
from nltk.tokenize import sent_tokenize


keyword_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/keywords.json"
title2dailycare = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/title2dailycare.csv"

all_post = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

with open(keyword_path, "r") as f:
    keywords = json.load(f)
    dailycare_keywords = keywords["dailycare"]
    psychosocial_keywords = keywords["psychosocial"]

dailycare_keyword = [str.lower(i) for i in dailycare_keywords]
psychosocial_keyword = [str.lower(i) for i in psychosocial_keywords]

with open(all_post, "r") as post:
    all_posts = json.load(post)

df = pd.read_csv(title2dailycare)
dailycare_title = list(df.loc[df["Final.7.types"] == "Daily care"].Title)
psychosocial_title = list(df.loc[df["Final.7.types"] == "Psychosocial"].Title)
no_relevant = list(df.loc[(df["Final.7.types"] != "Psychosocial") & (df["Final.7.types"] != "Daily care")].Title)
no_relevant_index = list(df.loc[(df["Final.7.types"] != "Psychosocial") & (df["Final.7.types"] != "Daily care")].Document)
dailycare_text = {}
psychosocial_text = {}
no_relevant_text = {}

title_index = {}

for i in all_posts:
    if i["title"] in dailycare_title:
        dailycare_text[i["title"]] = i["text"].replace("\n", "")
        indx = dailycare_title.index(i["title"])
        title_index[str(no_relevant_index[indx])] = i["text"]
    elif i["title"] in psychosocial_title:
        psychosocial_text[i["title"]] = i["text"].replace("\n", "")

    elif i["title"] in no_relevant:
        no_relevant_text[i["title"]] = i["text"].replace("\n", "")
        # indx = no_relevant.index(i["title"])
        # title_index[str(no_relevant_index[indx])] = i["title"]
print(title_index)

# dailycare_text = pd.DataFrame(dailycare_text)


# with open("dailycare_all_annotation.csv", "w", encoding="utf-8") as csvfile:
#     w = csv.writer(csvfile)
#     w.writerow(["sentences","label","title"])
#     for title,text  in dailycare_text.items():
#         w.writerow((text, "No-relevant", title))

with open("../ADRDtask2/other_test.csv", "w", encoding="utf-8") as csvfile:
    w = csv.writer(csvfile)
    for title, text in title_index.items():
        print(title)
        texts = sent_tokenize(text)
        for item in texts:
            if item =="" or len(item.split(" ")) <4:
                continue
            else:
                w.writerow((item, 0, title))



