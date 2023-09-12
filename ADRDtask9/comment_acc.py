import os
import pandas as pd
pred = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask9/pred.txt"

test_label = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/text_label.tsv"
test_title = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask8/text_title.csv"
df_label = pd.read_csv(test_label,sep = "\t")
df_title = pd.read_csv(test_title,sep = "\t")
label = []
with open(pred, "r") as f:
    for i in f.readlines():
       label.append(i.strip().split(" ")[-1])


df = pd.DataFrame({"label": label})
df.to_csv("pred.csv", index=False)
