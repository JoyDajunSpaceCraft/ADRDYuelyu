import json

post = "post path is a json file"
comment = "comment path is a json file"

with open(post, "r") as posts:
    po = json.load(posts)

with open(comment, "r") as comments:
    co = json.load(comments)

id2post = {} # id is key and value is [title and text]
id2comment = {}
for p in po:
    id = p["post_id"]
    id2post[id] = [p["title"], p["text"]]
for c in co:
    # "https://www.reddit.com/r/Alzheimers/comments/1d7lws/gps_tracking_for_alzheimers_patient_in_greece/cat2cwz/"
    # split link
    links = c["link"]
    post_id = links.split("comments/")[1].split("/")[0]
    if post_id not in id2comment.keys():
        id2comment[post_id] = [c["text"]]
    else:
        id2comment[post_id].append(c["text"])

all_pc = {}

for key, value in id2post.items():
    sub_all_pc = {}
    if key in id2comment.keys():
        sub_all_pc["title_text"] = value
        sub_all_pc["comment"] = id2comment[key]
        all_pc[key] = sub_all_pc

with open("all_post_comment.json", "w") as pc:
    json.dump(all_pc,pc)






