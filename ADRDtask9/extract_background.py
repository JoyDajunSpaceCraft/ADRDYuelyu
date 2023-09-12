import os
import json
path = "/Users/yuelyu/Downloads/GARD_qdecomp.v1"
import pandas as pd

def process_background_context():
    # sub_path = os.path.join(path, "GARD_Q_0001.ann")
    # ann
    dir_path = os.listdir(path)

    test_sentence = []
    test_label = []
    for i in dir_path:

        i = str(i)
        if i.endswith("ann"):
            new_path = os.path.join(path, i)
            with open(new_path,"r") as f:
                for i in f.readlines():
                    if "Background" in i.split("\t")[1]:
                        sentence = i.split("\t")[2]
                        test_sentence.append(sentence.strip())
                        test_label.append(0)
                    elif "Question" in i.split("\t")[1]:
                        sentence = i.split("\t")[2]
                        test_sentence.append(sentence.strip())
                        test_label.append(1)
                    else:
                        continue


    df = pd.DataFrame({"sentence":test_sentence, "label":test_label})
    df.to_csv("Clinical_test.tsv", index=False, sep="\t")





        # for i in f.readlines():
        #     print(i)


        
def check_overlap():
    post_200 = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/200_title_post_class.json"
    post_comment = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_post_comment.json"

    with open(post_200, "r") as f1:
        post_200_ = json.load(f1)
    with open(post_comment,"r") as f2:
        pc = json.load(f2)

    title_200 = []
    title_pc = []

    for key, value in post_200_.items():
        title_200.append(key)

    for key, value in pc.items():
        title = value["title_text"][0]
        title = str.lower(title)

        title_pc.append(title)

    for i in title_pc:
        if i in title_200:
            print(i)



if __name__ == "__main__":
    process_background_context()