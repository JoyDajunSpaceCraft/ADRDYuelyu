import pandas as pd
from nltk  import sent_tokenize
path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/all_new_modify_csv.csv"

def splite_sentence():
    df = pd.read_csv(path)
    all_sent = df["modify_text"].to_list()
    index_list = []
    sent_list = []
    for idx, i in enumerate(all_sent):
        tokenset = sent_tokenize(i)
        sent_list.extend(tokenset)
        index_list.extend([idx]* len(tokenset))
    print(sent_list[:10])
    print(index_list[:10])

    new_dict = {}
    new_dict["sentence"] = sent_list
    new_dict["post_index"] = index_list
    new_dict["label"] = [0] * len(index_list)
    new_df = pd.DataFrame(new_dict)
    new_df.to_csv("test_.csv",index=False)




def change():
    df = pd.read_csv("test_.csv")
    post2sent = {}
    with open("pred.txt") as f:
        for idx, i in enumerate(f.readlines()):
            i = i.split("\n")[0]
            if i.split(" ")[-1] == "1":
                sent = df.iloc[idx].sentence

                post_id = df.iloc[idx].post_index
                if post_id in post2sent.keys():
                    post2sent[post_id] = " ".join([post2sent[post_id], sent])
                else:
                    post2sent[post_id] = sent

    sent_list = []
    for i in sorted(post2sent):
        sent_list.append(post2sent[i])

    new_df = pd.DataFrame({"selfbody":sent_list})
    new_df.to_csv("dailycare_.csv", index=False)




change()