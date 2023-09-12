
import json
import pandas as pd
# this file is to generate the file with 200 categories items that send into the models
from nltk import sent_tokenize


all_path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

# this function to make the 200 posts with format
# {"title": ["sentence", label]}
def get_all_posts():
    # all_post = "all_posts.json"
    with open(all_path, "r", encoding="utf-8") as all_:
        json_file = json.load(all_)

    title2post_class = {}

    for i in json_file:
        title = reformat_str(i["title"])
        text = reformat_str(i["text"])
        if text != "":
            post_class = [text, "None"]
            title2post_class[title] = post_class

    # print(title2post_class)
    with open("all_post_title_post_class.json","w", encoding="utf-8") as cate_json:
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


def token_2post2sentence():
    json_file = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask5/all_post_title_post_class.json"
    with open(json_file, "r", encoding="utf-8") as f:
        title_sentence = json.load(f)

    sentences = []
    labels = []
    title_list = []
    hiw_labels = []

    for title, [post, label] in title_sentence.items():
        for i in sent_tokenize(post):
            sentences.append(i.strip())
            labels.appaend(0)
            title_list.append(title)
            hiw_labels.append(label)

    assert len(sentences) == len(labels) ==len(title_list) ==len(hiw_labels)
    dict = {"sentence":sentences,"label":labels}
    df = pd.DataFrame(dict)
    df.to_csv("Clinical_test.tsv",sep="\t", index=False)

    df2 = pd.DataFrame({"title":title_list, "sentence":sentences, "HIWlabel":hiw_labels})
    df2.to_csv("all_post_title_sentence_hiw.tsv",sep="\t", index=False)




def count_post():
    count = 0
    all_post = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"
    with open(all_post, "r", encoding="utf-8") as all_:
        json_file = json.load(all_)
    sent_count = 0
    sent_dict = {}
    for i in json_file:
        if i["text"] !="" and type(i["text"]) == str:
            # print(i["text"])
            count+=1

            sent_count +=len(sent_tokenize(i["text"]))
            if len(sent_tokenize(i["text"])) not in sent_dict.keys():
                sent_dict[len(sent_tokenize(i["text"]))] = 1
            else:
                sent_dict[len(sent_tokenize(i["text"]))]+=1
        # break
    print("we have {} of proper posts".format(count))
    print("the mean {} of mean".format(sent_count/count))
    print("the mediam number of all post is {}".format(sent_dict))

# def count_sentence_length():


if __name__ == "__main__":
    # get_all_posts()
    # token_2post2sentence()
    count_post()
    # token_200post2sentence()
