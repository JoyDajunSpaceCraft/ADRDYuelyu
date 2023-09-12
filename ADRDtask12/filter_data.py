import os
import pandas as pd
import praw
import re
import json


class CleanData():
    def __init__(self):
        self.search_item = ["alzheimer", "dementia"]
        self.bp = ["legaladvice", "Advice", "offmychest", "CaregiverSupport", "CaregiverSupport", "AmltheAsshole"]
        self.client_id = "pV-DoxK3yzaD8eJTuMBRWQ"
        self.client_secret = "dQvLvNze_-hu6DRhIU_Nsu6Od12E3A"
        self.username = "yuj49"
        self.password = "Fiona@1999"
        self.user_agent = "yuj49"
        self.non_adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_other_cats"
        self.adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_other_cats"
        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.client_secret,
                                  username=self.username,
                                  password=self.password,
                                  user_agent=self.user_agent)

    def clean_post(self):
        # non adrd
        posts = []
        comments = []
        urls = []
        non_list = os.listdir(self.non_adrd_path)
        non_list.remove(".DS_Store")
        for i in non_list:
            print(i)
            with open(os.path.join(self.non_adrd_path, i), "r") as f:
                json_ = json.load(f)
                for i in json_:
                    if "selftext" not in i.keys():
                        continue

                    this_post = self.re_lower_removehttp(i["selftext"])
                    this_post = self.re_lower_removehttp(i["selftext"])

                    if "alzheimer" in this_post or "dementia" in this_post :
                        posts.append(this_post)
                        urls.append(i["url"])
                        comments.append(" ")

        df = pd.DataFrame({"submission": posts, "comment": comments, "url":urls})
        df.to_csv("non_adrd.csv", index=False)
        # adrd data
        adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/all.csv"
        adrd = pd.read_csv(adrd_path)
        urls = adrd.url.values.tolist()
        groups = self.get_group_from_url(urls)
        adrdgroup_idx = [idx for idx, i in enumerate(groups) if str.lower(i) == "alzheimersgroup"]
        adrd = adrd.drop(index = adrdgroup_idx)
        print(adrd)
        adrd.to_csv("all.csv", index=False)

        posts = []
        comments = []
        urls = []
        inside_path ="/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_inside_cat"
        inside_list = os.listdir("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_inside_cat")
        inside_list.remove(".DS_Store")
        for i in inside_list:
            print(i)
            with open(os.path.join(inside_path, i), "r") as f:
                json_= json.load(f)
                for i in json_:
                    if "selftext" not in i.keys():
                        continue
                    this_post = self.re_lower_removehttp(i["selftext"])
                    posts.append(this_post)
                    urls.append(i["url"])
                    comments.append(" ")
        df = pd.DataFrame({"submission": posts, "comment": comments, "url": urls})
        df.to_csv("all.csv", index=False)

    def count_subreddit(self):
        searchs = ["alzheimers", "dementia", "alzheimerssupport", "alzheimerscanada", "alzheimersgroup"]
        adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/all.csv"
        adrd = pd.read_csv(adrd_path)
        urls = adrd.url.values.tolist()
        groups = self.get_group_from_url(urls)
        counts = {i: 0 for i in searchs}
        for i in searchs:
            for j in groups:
                if i == str.lower(j):
                    counts[i] += 1
        print(counts)

    def get_group_from_url(self, urls):
        groups = []
        for i in urls:
            groups.append(i.split("https://www.reddit.com/r/")[1].split("/")[0])
        return groups

    def get_id_from_url(self, urls):
        ids = []
        try:
            for i in urls:
                ids.append(i.split("https://www.reddit.com/r/")[1].split("/")[2])
        except Exception as e:
            print(urls)
        return ids

    def re_lower_removehttp(self, s):
        s = str.lower(s)
        s = re.sub(r'http\S+', '', s)
        s = re.sub(r'\n', '', s)
        return s

    def get_comment(self):
        adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/adrd.csv"
        non_adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/non_adrd.csv"
        df_adrd = pd.read_csv(adrd_path)
        df_non_adrd = pd.read_csv(non_adrd_path)
        base_ = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/adrd_all"
        for df, path in zip([df_adrd, df_non_adrd], ["adrd.csv", "non_adrd.csv"]):
            for idx, i in enumerate(df.comment.to_list()):
                if type(i) == float or i == " " or len(i.split(" ")) < 3:
                    submission_url = str(df.iloc[idx].url)
                    if "https://www.reddit.com/r/" not in submission_url:
                        continue
                    try:
                        submission_id = self.get_id_from_url([submission_url])[0]
                        submissions = self.reddit.submission(submission_id)
                        title = submissions.title
                        print("title here is: ", title)
                        comment_ids = submissions.comments._comments
                        comments = ""
                        for i in comment_ids:
                            comment = self.reddit.comment(str(i))
                            comments_ = comment.body
                            comments_ = self.re_lower_removehttp(comments_)
                            comments = " " + comments_
                        df.loc[idx, "comment"] = comments
                        df.loc[idx, "title"] = title
                    except Exception as e:
                        print(e)
                        print(idx)
                        continue
                # print(df.iloc[idx].comment)


                if idx %100==0 and idx!=0:
                    new_df = df.iloc[idx - 100:idx]
                    print(new_df)
                    limit = idx /100
                    new_df.to_csv(os.path.join(base_, str(limit) + "_" + path), index=False)
            # df.to_csv(os.path.join(base_, str(limit) + "_" + path), index=False)

    def clean_empty_post_comment(self):
        adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/adrd.csv"
        non_adrd_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/non_adrd.csv"
        df_adrd = pd.read_csv(adrd_path)
        df_non_adrd = pd.read_csv(non_adrd_path)
        for df, path in zip([df_adrd, df_non_adrd], ["adrd.csv", "non_adrd.csv"]):
            urls = df.url.values.tolist()
            comments = df.comment.values.tolist()
            posts = df.submission.values.tolist()
            for idx,(url, comment, post) in enumerate(zip(urls, comments, posts)):
                if "https://www.reddit.com/r/" not in url:
                    df = df.drop(index = idx)
                # if len(comment.split(" "))<3 and post in ["[removed]", "[deleted]", " ",""]:
                #     df = df.drop(index = idx)
            df.to_csv(path, index=False)


if __name__ == "__main__":
    c = CleanData()
    c.get_comment()
    # c.count_subreddit()
