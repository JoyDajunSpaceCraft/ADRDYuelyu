import json
import torch
max_padding_len = 395
with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/Oct_31_remove_punc/train_attribution.json", "r") as f:
  j = json.load(f)
  match_batch = []
  for idx, i in enumerate(j):
    tmp = []
    for x in i:
      tmp.append(x[1])
    if len(tmp) < max_padding_len:
      tmp.extend([1] * (max_padding_len - len(tmp)))

    # tmp = torch.FloatTensor([tmp])
    # print(tmp.shape)
    if idx!=0:
      match_batch.append(tmp)
    else:
      match_batch = [tmp]
match_batch = torch.FloatTensor(match_batch)
print(match_batch.shape)
