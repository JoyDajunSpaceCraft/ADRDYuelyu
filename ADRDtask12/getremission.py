import requests as requests
import datetime
import time
import json

# subreddit_name = "Alzheimers" #7339, from 2022-07-19 to 2010-08-26

# subreddit_name = "dementia" # 11880 posts, from 2022-07-19 to 2010-01-10
# subreddit_name = "AgingParents" # 3525 posts, from 2022-07-19 to 2014-04-29
# subreddit_name = "AlzheimersSupport" # 141 posts, from 2022-03-23 (newest post date) to 2012-10-12
# subreddit_name = "AlzheimersCanada"  # 240 posts, from 2015-04-13 to 2022-06-08 (newest post date)
# subreddit_name = "CaregiverSupport" #5328 posts, from 2014-01-31 to 2022-07-10
#"TrueOffMyChest","mentalhealth", "GriefSupport", "depression",
subreddit_name = ["Alzheimers", "dementia", "AlzheimersSupport","AlzheimersCanada"]
before_time = int(datetime.datetime.now().timestamp())

post_ids = set()
result = []

while True:
    previous_total = len(post_ids)
    print(len(post_ids))
    for i in subreddit_name:
        res = requests.get("https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=created_utc&size=100&before={time}".format(subreddit=i, time=before_time))
        try:
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
        except Exception as e:
            continue

    json.dump(result,open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask12/data_inside_cat/{0}.json".format(i),"w"),indent=2)