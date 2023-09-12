import pandas as pd
import numpy as np
df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/new_all_adrd.csv")
# new_df = df.loc[~df["selftext"].isna()]
# new_df.to_csv("new_all_adrd.csv", index=False)
print(len(df))