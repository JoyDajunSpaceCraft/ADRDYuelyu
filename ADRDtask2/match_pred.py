import json
import os
import pandas as pd
import numpy as np


original_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask2/other_test.csv"
index_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask2/" \
             "" \
             "'''.json"
pred_result = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask2/zero_shot_pred.txt"
res_list = []
with open(pred_result,"r") as predict:
    res = predict.readlines()
    res = res[1:]

    for index, item in enumerate(res):
        res_list.append(item.strip().split(" "))

# with open(index_path, "r") as title_index:


# get sentences from the 200 posts and split by the sentence value
# also need the id of the title
df = pd.read_csv(original_path)
question = {}

for index, _, label in res_list:
    if label=="1":
        index = int(index)
        title_index = df.iloc[index-1].title
        title_index = str(title_index)
        if title_index not in question.keys():
            question[title_index] = [df.iloc[index-1].sentences]
        else:
            question[title_index].append(df.iloc[index-1].sentences)
print(question)

final_question = []
for key, value in question.items():
    for j in value:
        final_question.append([key, j])
final_question = np.array(final_question)

# with open("question_title.json","w", encoding="utf-8") as question_title:
#     json.dump(question,question_title)


write_in = pd.DataFrame(final_question)

write_in.to_csv("qustion_title.csv", index=False)


print(df.iloc[1])






