import pandas as pd
import streamlit as st
import json
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/bart_new_order.csv"
df = pd.read_csv(path)

path_1760 = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/dailycare_1760_posts.csv"
df_1760 = pd.read_csv(path_1760)

posts = df.shorten_post.values.tolist()
hiw = df.hiw.values.tolist()
#Sure, here's a Python script that implements a simple BERT embedding search using the Hugging Face Transformers library.



path_large = ""

with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/refine_middle_1760_bart.json","r") as f:
    json_ = json.load(f)

for key, values in json_.items():
    # the format of the
    key = "rgk9zg" # normally venting posts id g7h6w3 fm28zd rgk9zg
    values = json_[key]
    post_ = df_1760.loc[df_1760.reddit_id==key].post.values[0]
    res = []
    color_loop = ["#faa","#8ef","#afa"]

    # for hiw_item, context in values["valid_context"].items():
    # for hiw_item, context in values["valid_context"].items():
    count_context = 0
    max_index = 0
    for key_idx in values["valid_context"].keys():
        if count_context< len(values["valid_context"][key_idx]):
            count_context = len(values["valid_context"][key_idx])
            max_index = key_idx

    hiw_item =max_index

    context = values["valid_context"][hiw_item]

    sents = sent_tokenize(post_)
    color_count = 0
    for idx, i in enumerate(sents):
        if i == hiw_item:
            res.append((i, "information want","#8ef"))
            color_count += 1
            print(color_count)
        elif i in context:
            res.append((i, "context","#faa"))
        else:
            res.append(i)

    for item in res:
        annotated_text(item)
        # break
    break
        # annotated_text(
        #     "My grandmother has dementia, I’m not 100% sure as I don’t think anyone has officially had her tested, it’s just “dementia”. I don’t think my family really understood until recently how bad it was.  From my research, she’s a late stage 5, early stage 6 Alzheimer’s.   She’s been “a bit forgetful” for about 10 years now, but in recent years we learnt that she doesn’t shower, doesn’t cook, has regular accidents etc. when she’s stressed or anxious she has difficulty with names/faces, but is generally good with her family or people she’s known a long time.   My grandfather recently passed quite suddenly and we had to move her into an aged care facility as she can’t live by herself.  This also revealed to us how much she relied on him to not look sick.   Her condition has significantly worsened since being there, she’s always confused, upset, angry. It’s not a dementia friendly place at all and the nurses aren’t the best at handling it. There’s no cues as to what time of day it is, the building is confusing and hard to find the dining/lounge room, etc.   She has regularly said that she may as well not exist anymore as she has nothing to live for and she’s mad at the family for dumping her there. A couple of times I have found notes in her room with her name, my grandfathers name, their address and phone number, kids names, and usually the words “help” or “take me home”.   My parents have a “guest house” that they currently use as an air bnb, and are considering moving her in there. From my mum and my research, it’s a dementia friendly space and there are slight modifications we can make to make it more enabling for her.  We’d also get home help, a morning or overnight nurse etc.   We could bring her furniture from home, give her simple tasks, take her out for activities.   More important than that is that she’d be around family more, have more independence and get her self confidence back. From my research she’s a late stage 5, early stage 6 Alzheimer’s   We’re aware that as she moves through stage 6 and into stage 7, she will get worse and will need to go into a dementia specific care place. ",
        #     ("Is it worth moving her now, to my parents, only to move her again in a year or so?", "information want"),
        #     "If she does move in to my parents (it’s actually a separate unit, self contained but with easy access to their house), what extra care would she need that we might have overlooked.   Or should we look at sending her straight to a memory care unit? ")
