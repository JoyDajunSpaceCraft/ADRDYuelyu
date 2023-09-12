import pandas as pd
import math
def write_sentence():
    path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/Alzheimers.csv"
    df = pd.read_csv(path)
    sentence = []
    label = []

    for i in list(df.selftext):
        if type(i)!=str and math.isnan(i):
            continue
        else:

            cleans = i.split("\n")
            new_clean = []
            for m in cleans:
                if "\t" in m:
                    continue
                else:
                    new_clean.append(m)
            i = " ".join(new_clean)
            if len(new_clean)> 1:
                sentence.append(i)
            else:
                sentence.append(new_clean[0])
            label.append(0)
    new_df = pd.DataFrame({"sentence":sentence[:1000],"label":label[:1000]})
    new_df.to_csv("text.tsv", sep="\t", index = False)


def map_pred():
    dfs = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask10/text.tsv", sep="\t")
    
    labels = ["None",'wandering', 'incontinence',
        'paranoia', 'shower', 'bath', 'hygiene', 'sleep',
        'eat', 'feeding', 'anger', 'agitation',
        'activities of daily living',
         'daily activity', 'activity',
         'driving', 'drive', 'safety', 'safe', 'supervision',
          'at home alone', 'clean', 'busy', 'interest',
           'leave', 'left', 'track', 'find', 'snack', 'cabinet',
            'lock', 'home', 'house', 'live', 'living', 'voicemail',
            'management', 'manage', 'control', 'care',
            'daily activities or caregiver', 'caregiving']
    
    preds = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask10/pred.txt"
    pred = None
    with open(preds, "r") as f:
        pred = f.readlines()
    val_s = []
    for idx, value in enumerate(pred):
        val = value.strip().split(" ")[-1]
        cat = int(val)
        val_s.append(labels[cat])
    sentences = list(dfs.sentence)
    new_df = pd.DataFrame({"sentence": sentences, "label": val_s})
    new_df.to_csv("pred_text.csv", index=False)

        

if __name__ == "__main__":
    map_pred()