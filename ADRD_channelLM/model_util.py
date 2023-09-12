import numpy as np
import os
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel

def load_checkpoint(gpt2, checkpoint):
    pass

def get_optimizer_and_scheduler():
    pass

def get_dataloader(inputs, batch_size, is_training):
    shape = inputs["input_ids"].shape
    for v in inputs.values():
        assert v.shape == shape

    if "labels" in inputs:
        # tensors that have the same size of the first dimension.
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"],
                                inputs["labels"])

    else:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"])

    if is_training:
        # Samples elements randomly. If without replacement,
        # then sample from a shuffled dataset. If with replacement,
        # then user can specify num_samples to draw.
        sampler = RandomSampler(dataset)

    else:
        # Samples elements sequentially, always in the same order.
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, smapler=sampler, batch_size=batch_size)
    return dataloader

def get_optimizer_and_scheduler(optimizer_name,
                                model,
                                learning_rate=1e-5,
                                warmup_proportion=0.01,
                                warmup_steps=50,
                                weight_decay=0.0,
                                adam_epsilon=1e-8,
                                num_training_steps=1000):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params':[p for n, p in model.named_parameters() if not any (nd in n for nd in no_decay)],'weight_decay':weight_decay},
        {'params':[p for n, p in model.named_parameters() if any (nd in n for nd in no_decay)],'weight_decay':0.0}
    ]
    if optimizer_name == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters,
                              lr=learning_rate,
                              eps=adam_epsilon) # Regularization constants for square gradient and parameter scale respectively

        # 由于刚开始训练时, 模型的权重(weights)
        # 是随机初始化的，此时若选择一个较大的学习率, 可能带来模型的不稳定(
        #     振荡)，选择Warmup预热学习率的方式，可以使得开始训练的几个epoches或者一些steps内学习率较小, 在预热的小学习率下，模型可以慢慢趋于稳定, 等模型相对稳定后再选择预先设置的学习率进行训练, 使得模型收敛速度变得更快，模型效果更佳
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_training_steps)

    else:
        raise NotImplementedError()
    return optimizer, scheduler


class MyEmbedding(torch.nn.Module):
    # for prompt tuning
    def __init__(self, embed, n_prefix):
        super().__init__()
        self.embed = embed
        self.new_embed = torch.nn.Embedding(n_prefix, embed.embedding_dim)

        indices = np.random.permutation(range(5000))[:n_prefix] # Randomly permute a sequence, or return a permuted range.
        init_weight = self.embed.state_dict()["weight"][indices] # the state_dict contain weight and bias
        self.new_embed._load_from_state_dict({"weight":init_weight},"",None, True,[],[],"") # copy and buffer the state dict in this model

    def forword(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight, self.new_embed.weight], 0),
            # the 0 here is append with row original is weight row *col now is (weight row + new weigth row)* weight col
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse
        )

class MyEmbedding2(torch.nn.Module):

    def __init__(self, embed, mapping):
        super().__init__()
        self.my_embed = torch.nn.Embedding(len(mapping), embed.embedding_dim)
        indices = [mapping[i] for i in range(len(mapping))]
        init_weight = embed.state_dict()["weight"][indices]
        self.my_embed._load_from_state_dict({"weight": init_weight},
                                            "", None, True, [], [], "")

    def forward(self, input):
        return self.my_embed(input)


class MyLMHead(torch.nn.Module):
    def __init__(self, lm_head, mapping):
        super().__init__()
        self.my_lm_head = torch.nn.Linear(lm_head.in_features, len(mapping), bias=False)

        indices = [mapping[i] for i in range(len(mapping))]
        init_weight = lm_head.state_dict()["weight"][indices]
        self.my_lm_head._load_from_state_dict({"weight": init_weight},
                                              "", None, True, [], [], "")
    # TODO check the place the head LM is add another weight these weight is from reassign_output_tokens
    def forward(self, input):
        return self.my_lm_head(input)
        

class MyLMHeadWithTransform(torch.nn.Module):

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
        self.transform = torch.nn.Linear(lm_head.in_features,
                                         lm_head.in_features, bias=False)
        init_weight = torch.eye(lm_head.in_features)
        self.transform._load_from_state_dict({"weight": init_weight},
                                              "", None, True, [], [], "")

    def forward(self, input):
        return self.lm_head(self.transform(input))

def set_extra_embeddings(model, n_prefix):
    model.transformer.set_input_embeddings(
        MyEmbedding(model.transformer.wte, n_prefix))

def set_separate_lm_head(model, mapping):
    model.set_output_embeddings(
        MyLMHead(model.lm_head, mapping))

def set_separate_embeddings(model, mapping):
    model.set_input_embeddings(
        MyEmbedding2(model.transformer.wte, mapping))

def set_transformed_lm_head(model):
    model.set_output_embeddings(
        MyLMHeadWithTransform(model.lm_head))