import pandas as pd
from nltk import sent_tokenize
df = pd.read_csv("/ADRDtask7/updated_aggrement_calculation_86.xlsx - 工作表 1.tsv")
context = df["context"]
context = list(context)
sentence = df["question"]
sentence = list(sentence)
question_list = []
background_list = []
for idx, value in enumerate(sentence):
    try:
        if "?" in value and len(value) - value.index("?") > 2:
            question = value.split("?")[0] + "?"
            background = " ".join(value.split("?")[1:])
        else:
            values = sent_tokenize(value)
            question = values[0]
            background = " ".join(values[1:])
    except Exception as e:
        print(value)
    question_list.append(question)
    background_list.append(background)
    print(question_list)
    print(background_list)

newdf = pd.DataFrame({"context": context, "question":question_list, "background":background_list})
newdf.to_csv("exact_partial_.csv", index=False)