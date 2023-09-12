import pandas as pd
import praw
path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/get_1736_comment/dailycare_1736_post.csv"
original = pd.read_csv(path)
sumbission_id = original.id.values.tolist()
client_id = "pV-DoxK3yzaD8eJTuMBRWQ"
client_secret = "dQvLvNze_-hu6DRhIU_Nsu6Od12E3A"
username = "yuj49"
password = "Fiona@1999"
user_agent = "yuj49"
reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  username=username,
                                  password=password,
                                  user_agent=user_agent)
comments = []

for i in sumbission_id:
    with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/get_1736_comment/dailycare_1736_comment/{}.txt".format(i), "w") as f:
        comment_queue = reddit.submission(i).comments[:]
        # count = 0
        try:
            while comment_queue:
                comment = comment_queue.pop(0)
                f.write(comment.body)
                f.write("\n")
                comment_queue.extend(comment.replies)
                # count +=1
        except Exception as e:
            print("wrong number is ", i)


# original["num_comments"] = counts
# original.to_csv("new_count_comment.csv", index=False)
    # print(len(j.body))
    # comment_queue = submission.comments[:]
    # for j in s:
    #     print(j.body)


