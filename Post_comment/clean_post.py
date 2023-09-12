import json
import pandas as pd
# this file is to generate the file with 200 categories items that send into the models
from nltk import sent_tokenize

# this function to make the 200 posts with format
# {"title": ["sentence", label]}
def get_200_posts():
    all_post = "all_posts.json"
    with open(all_post, "r", encoding="utf-8") as all_:
        json_file = json.load(all_)

    annotation_200 = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/title2dailycare.csv"
    annotations = pd.read_csv(annotation_200)
    final_types = list(annotations["Final.7.types"])
    titles_in200 = list(annotations["Title"])
    reformat =lambda i: reformat_str(i)
    type_list = [reformat(i) for i in final_types]
    title_list = [reformat(i) for i in titles_in200]


    # print(title_list)

    title2post_class = {}

    for i in json_file:
        title = reformat_str(i["title"])
        if title in title_list:
            idx = title_list.index(title)
            text = reformat_str(i["text"])
            if text == "":
                print(title) # because they don't contain any class inside so break
                continue
            post_class = [text, type_list[idx]]
            title2post_class[title] = post_class

    # print(title2post_class)
    with open("200_title_post_class.json","w", encoding="utf-8") as cate_json:
        json.dump(title2post_class,cate_json)

def reformat_str(strs:str):
    strs = strs.lower().strip()
    temp = strs.encode('ascii',"ignore")
    temp = temp.decode()
    return temp


# get answer by combine ## tokens
def summer_answer(tokens, answer_start, answer_end):
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]
    return answer


def token_200post2sentence():
    json_file = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/200_title_post_class.json"
    with open(json_file, "r", encoding="utf-8") as f:
        title_sentence = json.load(f)

    sentences = []
    labels = []
    title_list = []
    hiw_labels = []

    for title, [post, label] in title_sentence.items():
        for i in sent_tokenize(post):
            sentences.append(i.strip())
            labels.append(0)
            title_list.append(title)
            hiw_labels.append(label)

    assert len(sentences) == len(labels) ==len(title_list) ==len(hiw_labels)
    dict = {"sentence":sentences,"label":labels}
    df = pd.DataFrame(dict)
    df.to_csv("Clinical_test.tsv",sep="\t", index=False)

    df2 = pd.DataFrame({"title":title_list, "sentence":sentences, "HIWlabel":hiw_labels})
    df2.to_csv("title_sentence_hiw.tsv",sep="\t", index=False)




def count_post():
    count = 0
    all_post = "all_posts.json"
    with open(all_post, "r", encoding="utf-8") as all_:
        json_file = json.load(all_)
    for i in json_file:
        if i["text"] !="" and type(i["text"]) == str:
            # print(i["text"])
            count+=1
        # break
    print(count)

if __name__ == "__main__":
    # get_200_posts()
    # count_post()
    token_200post2sentence()