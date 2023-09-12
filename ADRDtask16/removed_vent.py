import pandas as pd
from nltk.tokenize import sent_tokenize
def generate_Ningfile():
    original_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/main_file.csv"
    not_sure = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/not_sure_memory_loss.csv"
    pink_memory = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/pink_memory_loss.csv"
    sure_memory = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/sure_memory_loss.csv"

    original_df = pd.read_csv(original_path)
    all_reddit_ids = original_df.reddit_id.values.tolist()
    Posts = original_df.Post.values.tolist()
    Titles = original_df.title.values.tolist()

    sure_df = pd.read_csv(sure_memory)
    sure_id = sure_df.reddit_id.values.tolist()
    print(len(sure_id))

    not_sure_df = pd.read_csv(not_sure)
    not_sure_df = not_sure_df.reddit_id.values.tolist()
    print(len(not_sure_df))

    pink_df = pd.read_csv(pink_memory)
    pink_id = pink_df.reddit_id.values.tolist()
    print(len(pink_id))

    rank_list = []
    posts = []
    titles = []
    # all_reddit_ids = list(set(all_reddit_ids))
    for idx, i in enumerate(all_reddit_ids):
        if i in sure_id:
            rank_list.append(1)
        elif i in not_sure_df:
            rank_list.append(2)
        elif i in pink_id:
            rank_list.append(3)
        else:
            rank_list.append(0)
        posts.append(Posts[idx])
        titles.append(Titles[idx])

    # dup_idx = {}
    # for idx, i  in enumerate(rank_list):
    #     if rank_list.count(i)>1:
    #         if i not in dup_idx.keys():
    #             dup_idx[i] = [idx]
    #         else:
    #             dup_idx[i].append(idx)

    # for key, value in dup_idx.items():
    #     for idx in value[:-1]:
    #         del posts[idx]
    #         del titles[idx]
    #         del rank_list[idx]


    assert len(posts) == len(titles) == len(rank_list)
    shorten = []
    for i, j in zip(posts, titles):
        questions = []
        for sent in sent_tokenize(" ".join([i, j])):
            sent = str.lower(sent)
            if ("how" in sent or "what" in sent or "why" in sent or "any" in sent) and "?" in sent:
                questions.append(sent)
            else:
                questions.append(" ")
        shorten.append(" ".join(questions))
    print(len(rank_list))
    print(len(posts))
    print(len(titles))
    print(len(shorten))
    new_df = pd.DataFrame({"reddit_id": all_reddit_ids, "post": posts, "title": titles,"hiw":shorten, "memory_loss":rank_list})
    new_df.to_csv("dailycare_1760_posts.csv", index=False)



def fine_tuning_vent():
    path_1760 = ""
if __name__ == "__main__":
    generate_Ningfile()

