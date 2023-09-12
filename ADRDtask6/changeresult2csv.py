import json
import pandas as pd
json_file = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask6/pred_result_new/xlnet.json"

with open(json_file, "r", encoding="utf-8") as j:
    json_file = json.load(j)
    context_list = []
    question_list = []
    background_list = []
    label_list = []
    for context, value in json_file.items():
        if value["answer"] is None or len(value["answer"].split()) <3:
            continue
        context_list.append(context)
        question_list.append(value["question"])
        background_list.append(value["answer"])
        label_list.append("daily care")
    df = pd.DataFrame({"context": context_list, "question": question_list, "background": background_list, "label": label_list})

    df.to_csv("case_study.csv", index = False)
