import json
import pandas as pd
import re
# path_json = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"
# new_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/dailycare_.csv"
# res = []
# with open(path_json, "r") as f:
#     j = json.load(f)
#     for i in j:
#         if i["text"] == "" or i["text"] == "[removed]":
#             continue
#         text = re.sub(r'\n', '', i["text"])
#         res.append(text)
#
# df = pd.read_csv(new_path)
#
# res.extend(df.selfbody.to_list())
# new_df = pd.DataFrame({"sentence":res, "label":[0]*len(res)})
#
# new_df.to_csv("text.tsv",sep = "\t", index=False)

pred = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/pred.txt"
tsv_ = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/text.tsv"
daily_care = []
df = pd.read_csv(tsv_, sep="\t")
with open (pred, "r") as f:
    for idx, i in enumerate(f.readlines()):
        s = i.split(" ")[-1].strip()
        if s == "1":
            daily_care.append(df.iloc[idx].sentence)
        print(daily_care)

new_df = pd.DataFrame({"sentence":daily_care})
new_df.to_csv("all_dailycare.csv", index=False)