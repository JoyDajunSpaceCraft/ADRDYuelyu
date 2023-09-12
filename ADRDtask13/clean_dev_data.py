import pandas as pd

df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask13/remove_dup.csv")
print(len(df))

df = df.drop_duplicates()
df.to_csv("remove_dup.csv", index=False)
