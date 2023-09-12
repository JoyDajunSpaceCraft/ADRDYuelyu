import requests
import pandas as pd
import praw
import time
import os
import json
from datetime import datetime

def offical_api():
    # note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
    auth = requests.auth.HTTPBasicAuth('pV-DoxK3yzaD8eJTuMBRWQ', 'dQvLvNze_-hu6DRhIU_Nsu6Od12E3A')

    # here we pass our login method (password), username, and password
    data = {'grant_type': 'password',
            'username': 'yuj49',
            'password': 'Fiona@1999'}

    # setup our header info, which gives reddit a brief description of our app
    headers = {'User-Agent': 'MyBot/0.0.1'}

    # send our request for an OAuth token
    res = requests.post('https://www.reddit.com/api/v1/access_token',
                        auth=auth, data=data, headers=headers)

    # convert response to JSON and pull access_token value
    print(res.json())
    TOKEN = res.json()['access_token']

    # add authorization to our headers dictionary
    headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

    # while the token is valid (~2 hours) we just add headers=headers to o
    # params = {'limit': 200}
    df = pd.DataFrame()
    before_item = "t3_u3uvl1"
    df = df.append({
        'subreddit': None,
        'title': None,
        'selftext': None,
        'upvote_ratio': None,
        'created_utc': None,
    }, ignore_index=True)
    for i in range(1,3):
        params= {"before":before_item,"limit":100}
        # params = {"limit":100}
        # make a request for the trending posts in /r/Python
        res = requests.get("https://oauth.reddit.com/r/Alzheimers/new",params = params,
                           headers=headers)

        post = None
        tmp = res.json()['data']['children']
        for post in res.json()['data']['children']:
            # append relevant data to dataframe
            if post['data']['selftext'] in df["selftext"]: continue
            df = df.append({
                "author":post["data"]["author"],
                "name":post['data']["name"],
                'subreddit': post['data']['subreddit'],
                'title': post['data']['title'],
                'selftext': post['data']['selftext'],
                'upvote_ratio': post['data']['upvote_ratio'],
                'created_utc': datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
            }, ignore_index=True)
        print(df)
        before_item = post['data']["name"]

    df.to_csv("/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/Alzheimers1.csv", index=False)


def praw_scrapy(subname):
    # Authorized instance
    reddit_authorized = praw.Reddit(client_id="pV-DoxK3yzaD8eJTuMBRWQ",  # your client id
                                    client_secret="dQvLvNze_-hu6DRhIU_Nsu6Od12E3A",  # your client secret
                                    user_agent="yuj49",  # your user agent
                                    username="yuj49",  # your reddit username
                                    password="Fiona@1999",
                                    )

    subreddit = reddit_authorized.subreddit(subname)
    df = pd.DataFrame()

    df = df.append({
        'subreddit': None,
        'title': None,
        'selftext': None,
        'comment': None,
        'created_utc': None,
        "author":None
    }, ignore_index=True)

    for i in subreddit.new(limit=60000):
        comment = []
        for j in i.comments:
            comment.append(j.body)

        df = df.append({
            "name": i,
            'title': i.title,
            'selftext': i.selftext,
            "comment":"\n".join(comment),
            'created_utc': datetime.fromtimestamp(i.created_utc),
            "author":i.author
        }, ignore_index=True)

    df.to_csv("/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit/{}.csv".format(subname), index=False)



def caculate_posts(subname):
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    df = pd.read_csv(os.path.join(base_path, subname+".csv"))

    df['created_utc_'] = pd.to_datetime(df['created_utc']).astype(int)/ 10**9
    df.sort_values(by='created_utc_',ascending=False)
    print(df["created_utc"])
    ac = df[df['created_utc_'] > datetime.timestamp("2021-05-01").astype(int)/ 10**9]

def caculate_comment(subname):
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    count = 0
    df = pd.read_csv(os.path.join(base_path, subname + ".csv"))
    for i in df.comment:
        if str(i)!="nan":
            count+=len(i.split("\n"))
    print("{} have {} comments".format(subname, count))


def caculate_author():
    path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"
    with open(path, "r") as f:
        json_ = json.load(f)
    author_list = set()

    for i in json_:
        author_list.add(i["author"])
    base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    subnames = ["Alzheimers","Dementia","AlzheimersGroup","AlzheimersSupport","AgingParents"]
    for i in subnames:
        df = pd.read_csv(os.path.join(base_path, i+".csv"))
        for i in df.author:
            author_list.add(i)
    print(len(author_list))




if __name__ == "__main__":
    subreddits = ["Alzheimers","Dementia","AlzheimersGroup","AlzheimersSupport","AgingParents"]

    # for i in subreddits[2:]:
    # praw_scrapy("AlzheimersDisease")
    # offical_api()
    # for i in subreddits:
    #     caculate_comment(i)
    caculate_author()


# python src/subreddit_downloader.py AgingParents --batch-size 1000 --laps 5 --reddit-id pV-DoxK3yzaD8eJTuMBRWQ --reddit-secret dQvLvNze_-hu6DRhIU_Nsu6Od12E3A --reddit-username yuj49 --utc-after 1619900602
