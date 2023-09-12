import requests
import datetime
import time
import json

# how to fetch all posts from reddits:
# https://www.reddit.com/r/redditdev/comments/ce4fj3/best_way_to_get_all_posts_from_a_subreddit/

# Zhendong's version （up to May 2021）
# subreddit_name = "Alzheimers" #6997
# subreddit_name = "AlzheimersGroup" #3458
# subreddit_name = "AgingParents" #2085
# subreddit_name = "AlzheimersSupport" #63
# subreddit_name = "Alzheimers_Disease" #2
# subreddit_name = "AlzheimersCanada" #236
# subreddit_name = "dementia" #9076

# Ning's version (up to August, 2021)
# subreddit_name = "Alzheimers" #6207
# subreddit_name = "Dementia" #8924
# subreddit_name = "DementiaOntario" #25
# subreddit_name = "DementiaHelp" #27
# subreddit_name = "dementiaproducts" #15
# subreddit_name = "Alzheimers_Disease" #0
# subreddit_name = "AlzheimersGroup" #4850
# subreddit_name = "AlzheimersCanada" #233
# subreddit_name = "caregivers" #1090
# subreddit_name = "AlzheimersSupport" #89
# subreddit_name = "CaregiversSupport" #0
subreddit_name = "OpiatesRecovery"
before_time = int(datetime.datetime.now().timestamp())

post_ids = set()
result = []
while True:
    previous_total = len(post_ids)
    print(len(post_ids))
    res = requests.get(
        "https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=created_utc&size=100&before={time}".format(
            subreddit=subreddit_name, time=before_time))
    print(res)
    for post in res.json()["data"]:
        post_id = post["id"]
        if post_id not in post_ids:
            post_ids.add(post["id"])
            result.append(post)
            print(post_id)
            print(datetime.datetime.fromtimestamp(post["created_utc"]))
    if len(post_ids) == previous_total:
        break
    else:
        before_time = post["created_utc"]
        time.sleep(1)
json.dump(result, open("/Users/yuelyu/PycharmProjects/ADRD/Post_comment/{0}.json".format(subreddit_name), "w"), indent=2)