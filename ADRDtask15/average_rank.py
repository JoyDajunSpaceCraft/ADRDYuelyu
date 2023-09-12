import pandas as pd
import matplotlib.pyplot as plt

def plot_main():
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/new_Plot_basd.csv")

    bartIsmemoryloss_top50 = df.bartIsmemoryloss_top50.values.tolist()
    old_rank = []
    previous_idx = []
    for idx, i in enumerate(bartIsmemoryloss_top50):
        if i:
            previous_idx.append(idx+1)

    print(sum(previous_idx)/len(previous_idx))



plot_main()