import pandas as pd
import json
from nltk.tokenize import sent_tokenize
df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/new_all_adrd.csv")
ids = df.id.values.tolist()
post = df.selftext.values.tolist()
title = df.title.values.tolist()
title_post = []
for p,t in zip(post, title):
    title_post.append(" ".join([t, p]))

with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/30_template.json","r") as f:
    j = json.load(f)

    extra_post_category = ["daily care","psychosocial", ""]
    for id, value in j.items():
        t_p =  title_post[ids.index(id)]
        sents = 0
        # for k, v in value.items():
        if value["hiw"] ==[]:
            continue
        for m, n in value["valid_context"].items():
            sents += len(n)
        if len(sent_tokenize(t_p)) == sents:
            print(id)

        else:
            print(id)
            print("length in post", len(sent_tokenize(t_p)))
            print("length in extract", sents)
            # print("original title post is ",t_p)
            # print("extracted post is ",value["valid_context"])



