import os
import pandas as pd
import json
submission  = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/subreddit-comments-dl-master/dataset/20220517223201/submissions.csv"
older_post = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

df = pd.read_csv(submission)
ids = list(set(list(df.id)))
zhendong_ids = set()
with open(older_post,"r") as file:
    f_ = json.load(file)
    for i in f_:
        zhendong_ids.add(i["post_id"])


print(len(zhendong_ids))
print(len(ids))

count = 0
for i in ids:
    if i not in zhendong_ids:
        # print(i)
        count+=1
print(count)

