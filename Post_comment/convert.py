import json
import pandas as pd
path = "/Users/yuelyu/PycharmProjects/ADRD/Post_comment/all_posts.json"

from datetime import datetime
# title = []
# post = []
# category = []
with open(path, "r") as f:
    json_ = json.load(f)
create_time = []
for i in json_:
    create_time.append(i["create_time"])
create_time.sort()
date_time = datetime.fromtimestamp(create_time[-1])
print(date_time.strftime( '%Y-%m-%d %H:%M:%S'))
#
# for key, value in json_.items():
#     title.append(key)
#     post.append(value[0])
#     category.append(value[1])
#
# dic = {}
# dic = {"title":title,"post":post,"category":category}
# df = pd.DataFrame(dic,columns=['title', 'post','category'])
#
# df.to_csv("200_annotated.csv",index=False)