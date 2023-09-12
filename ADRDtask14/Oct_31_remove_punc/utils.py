import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import copy


test = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/test_memory_loss.csv"
train = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/train_memory_loss.csv"
valid = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/valid_memory_loss.csv"
test = pd.read_csv(test)
train = pd.read_csv(train)
valid = pd.read_csv(valid)


def match_post_title(df, name):
    pt = df.Post_title.values.tolist()
    idx = 0
    while type(pt[idx]) != float:
        idx += 1

    posts = df.iloc[idx:].Post.values.tolist()
    titles = df.iloc[idx:].title.values.tolist()

    for index, (i, j) in enumerate(zip(posts, titles)):
        pt[idx + index] = " ".join([i, j])
    df.Post_title = pt
    # print(pt)
    df.to_csv("{}.csv".format(name), index=False)

def remove_non_ascii_2(string):
    return string.encode('ascii', errors='ignore').decode()


def remove_strange(df, name):
    new_pt = []
    new_p = []
    new_t = []
    pt = df.Post_title.values.tolist()
    p = df.Post.values.tolist()
    t = df.title.values.tolist()


    for i, j, k in zip(pt, p, t):
        new_pt.append(remove_non_ascii_2(i))
        new_p.append(remove_non_ascii_2(j))
        new_t.append(remove_non_ascii_2(k))
    df.Post_title = new_pt
    df.Post = new_p
    df.title = new_t
    df.to_csv("{}.csv".format(name), index=False)


def count_similarity():
    from transformers import AutoTokenizer, AutoModel
    from nltk.tokenize import sent_tokenize
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    test_sent = "my 86y old grandma has been battling dementia for nearly 5 years now . it started with her repeating stories but it went downhill after my grandpa died . her short term memory is horrible . she will ask the same question many times , forgets my name sometimes ( but knows who we all are ) and generally acts like a child most of the time . she seems to always be very happy , doesn ' t understand tv shows but finds it amusing to watch and the most interesting thing - she ' s completely capable of living by herself . she lights fire in the fireplace , eats food we leave her and takes meds ( i have to call her to remind her of everything tho ) . we visit her 2 times a week . the only thing that ' s gotten progressively worse is her short term memory and even her long term but she still knows who we all are and understands everything we tell her . it ' s been a slow decline during these past 5 years . what to expect in the future ? what to expect in the future ? "

    training_data = "train_memory_loss.csv"
    df = pd.read_csv(training_data)
    train_data = df.Post.values.tolist()
    train_sents = []
    idx2sent = {}
    for idx, i in enumerate(train_data):
        idx2sent[idx] = sent_tokenize(i)
    import pandas as pd
    from nltk.tokenize import sent_tokenize
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import copy

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    target_sent = "my 86y old grandma has been battling dementia for nearly 5 years now . it started with her repeating stories but it went downhill after my grandpa died . her short term memory is horrible . she will ask the same question many times , forgets my name sometimes ( but knows who we all are ) and generally acts like a child most of the time . she seems to always be very happy , doesn ' t understand tv shows but finds it amusing to watch and the most interesting thing - she ' s completely capable of living by herself . she lights fire in the fireplace , eats food we leave her and takes meds ( i have to call her to remind her of everything tho ) . we visit her 2 times a week . the only thing that ' s gotten progressively worse is her short term memory and even her long term but she still knows who we all are and understands everything we tell her . it ' s been a slow decline during these past 5 years . what to expect in the future ? what to expect in the future ? "

    training_data = "/content/gdrive/MyDrive/ADRD/clustering/bert_classification/memory_loss/Oct_31_remove_punc/train_memory_loss.csv"
    df = pd.read_csv(training_data)

    train_data = df.Post_title.values.tolist()

    train_sents = []
    idx2sent = {}

    for idx, i in enumerate(train_data):
        idx2sent[idx] = sent_tokenize(i)
    new_simliar = None
    # def count_similar(target_sent, idx2sent, new_simliar):
    newsent = [target_sent]
    # the format is [target sent, : all other sent in training]
    for i in idx2sent.values():
        newsent.append(i[0])

    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in newsent:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,  # original 128
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    print("output keys", outputs.keys())

    embeddings = outputs.last_hidden_state
    print("last hiddent state", embeddings)

    attention_mask = tokens['attention_mask']
    print("attention shape is", attention_mask.shape)

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    print("mask shape", mask.shape)

    masked_embeddings = embeddings * mask
    print("mask shape embeddings", masked_embeddings.shape)

    summed = torch.sum(masked_embeddings, 1)
    print("sum shape", summed.shape)

    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    print("summed_mask shape", summed_mask.shape)

    mean_pooled = summed / summed_mask
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()

    simliar = cosine_similarity(
        [mean_pooled[0]],
        mean_pooled[1:]
    )
    max_value = max(simliar[0])
    new_simliar = copy.deepcopy(simliar[0])
    new_simliar.sort()

    for i in new_simliar[-20:]:
        print(i)
        index = np.where(simliar[0] == i)
        index = list(index)
        print(type(index))
        print(index[0][0])
        print(idx2sent[index[0][0]])

if __name__ =="__main__":
    # remove_strange(test,"test_memory_loss")
    # match_post_title(test,"test_memory_loss")
    # match_post_title(train,"train_memory_loss")
    # match_post_title(valid,"valid_memory_loss")

    count_similarity()
