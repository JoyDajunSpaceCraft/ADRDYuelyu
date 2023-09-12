import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging
import numpy as np
import csv

from tqdm import tqdm
from collections import Counter, defaultdict

from run import train, inference
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import prepend_task_tokens,reassign_output_tokens
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_separate_embeddings, set_transformed_lm_head

N_LABELS_DICT = {"ADRD": 3}


def load_data(data_dir, task, k, seed, split):
    # here split is define as test dev or train
    data_dir = os.path.join(data_dir, "k-shot", task, "{}-{}".format(k, seed))
    data = []
    if os.path.exists(os.path.join(data_dir, "{}.tsv".fotmat(split))):
        with open(os.path.join(data_dir, "{}.tsv".format(split)), "r") as f:
            for line in f:
                data.append(line.strip().split("\t"))
            data = data[1:]
    elif os.path.exists(os.path.join(data_dir, "{}.csv".format(split))):
        with open(os.path.join(data_dir, "{}.csv".format(split)), "r") as f:
            for sentence, label, title in csv.reader(f):
                data.append((sentence, label))
    else:
        raise NotImplementedError(data_dir)

    # make sure all of the data are in the format [input, output]
    assert np.all([len(dp) == 2 for dp in data])
    return data


def get_prompt(task, idx):
    if task == "ADRD":
        templates = ["This is about %s. ", "It is about %s.", "Topic: %s.", "Subject: %s. "]

        label_words = ["ADRD care at home", "Practical ideas", "No-relevant"]
    else:
        raise NotImplementedError(task)
    return [templates[idx] % label_word for label_word in label_words]


#
# class MyEmbedding(torch.nn.Module):
#     def __init__(self, embed, n_prefix):
#         super().__init__()
#         self.embed = embed
#         self.new_embed = torch.nn.Em


def get_paths(out_dir, gpt2, method, task, do_zeroshot,
              k, seed, train_seed, split, template_idx,
              batch_size=None, lr=None, warmup_steps=None,
              use_demonstrations=False,
              ensemble=False,
              prompt_tune=False,
              head_tune=False,
              transform_tune=False,
              n_prefix=20):
    model_name = gpt2

    if not do_zeroshot:
        if prompt_tune:
            model_name += "-prompt-ft"
            if n_prefix != 20:
                model_name += "-{}".format(n_prefix)
            elif head_tune:
                model_name += "-head-ft"
            elif transform_tune:
                model_name += "-transform-ft"
            else:
                model_name += "-all-ft"

        base_dir = os.path.join(out_dir, model_name,
                                "{}{}{}".format(method, "-demon" if use_demonstrations else "",
                                                "-ensemble" if ensemble else ""), task)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if do_zeroshot:
            cache_path = str(split)
            if use_demonstrations:
                cache_path += "-k={}-seed={}".format(k, seed)

            if use_demonstrations:
                cache_path += "-tseed={}".format(train_seed)

            cache_path += "-t={}".format(template_idx)

            return os.path.join(base_dir, cache_path + ".pkl")

        # if not the zeroshot means have pretrain procedure
        assert batch_size is not None and lr is not None and warmup_steps is not None

        out_dir = "BS={}-k={}-t={}-seed={}-tseed={}-lr={}{}".format(
            batch_size, k, template_idx, seed, train_seed, lr,
            "-wamrup={}".format(warmup_steps) if warmup_steps > 0 else "",
        )

        return os.path.join(base_dir, out_dir)


def prepare_data(tokenizer, train_data, test_data, max_length, max_length_per_example,
                 n_classes=2, templates=None, method_type="generative",
                 is_training=False, use_demonstrations=False,
                 ensemble=False, is_null=False):
    if type(templates) == list:  # because the templates is from get_prompt always return as a list
        transform = None
        assert len(templates) == n_classes
    else:
        transform = templates
    assert method_type in ["direct", "channel"]
    bos_token_id = tokenizer.bos_token_id  # begin token id
    eos_token_id = tokenizer.eos_token_id  # end token id

    if is_null:
        assert test_data is None
        assert method_type == "direct"
        test_data = [("N/A", "0")]
    prefixes_with_space = None
    if transform is None:
        templates = [template.strip() for template in templates]
        if method_type == "direct":
            templates = [" " + template for template in templates]
            if use_demonstrations:
                test_date = [(" " + sent, label) for sent, label in test_data]
        elif method_type == "channel":
            test_data = [(" " + sent, label) for sent, label in test_data]
            if train_data is not None:
                train_data = [(" " + sent, label) for sent, label in train_data]
            prefixes_with_space = [tokenizer(" " + template)["input_ids"] for template in templates]
        else:
            raise NotImplementedError()

    if transform is None:
        test_inputs = [tokenizer(sent)["input_ids"] for sent, _ in test_data]
        # get all of the input data length more than 256 - 16 (TODO why minus 16 here?)
        truncated = np.sum([len(inputs) > max_length_per_example - 16 for inputs in test_inputs])

        if truncated > 0:
            test_inputs = [inputs[:max_length_per_example - 16] for inputs in test_inputs]
            print("%d/%d truncated" % (truncated, len(test_inputs)))

        prefixes = [tokenizer(template)["input_ids"] for template in templates]  # get the inputsid
        idx = [idx for idx, _prefixes in enumerate(zip(*prefixes))
               if not np.all([_prefixes[0] == _prefix for _prefix in _prefixes])][
            0]  # TODO check the input id? make sure input id not only contain the prefix
    else:
        test_inputs = [transform(dp, tokenizer,
                                 max_length_per_example - 16,
                                 groundtruth_only=is_training)
                       for dp in test_data]
        if not is_training:
            assert np.all([len(dp) == 2 and
                           np.all([len(dpi) == n_classes for dpi in dp])
                           for dp in test_inputs])

    if is_training:
        assert not use_demonstrations
        assert not ensemble

        input_ids, attention_mask, token_type_ids = [], [], []
        for test_input, dp in zip(test_inputs, test_data):
            if transform is not None:  # TODO what is transform?
                test_input, test_output = test_input  # if use transform the
                encoded = prepro_sentence_pair_single(test_input, test_output, max_length, bos_token_id, eos_token_id)
            else:
                prefix = prefixes[int(dp[1])]  # TODO dp[1] -> test_data's label
                if method_type == "channel":
                    encoded = prepro_sentence_pair_single(prefix, test_input, max_length, bos_token_id, eos_token_id)
                elif method_type == "direct":
                    encoded = prepro_sentence_pair_single(test_input + prefix[:idx], prefix[idx:], max_length,
                                                          bos_token_id, eos_token_id)

                else:
                    raise NotImplementedError()
            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.apend(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))
    if use_demonstrations:
        if transform is not None:  # use demonstrations and transform is not happed together
            raise NotImplementedError()
        if ensemble:
            return prepare_data_for_parallel(
                tokenizer, train_data, test_data,
                max_length, max_length_per_example,
                method_type, n_classes,
                test_inputs, prefixes, idx, prefixes_with_space,
                bos_token_id, eos_token_id)

        assert train_data is not None
        demonstrations = []
        np.random.shufffle(train_data)

        for sent, label in train_data:
            if len(demonstrations) > 0:
                if method_type == "direct":
                    sent = " " + sent
                elif method_type == "channel":
                    prefixes = prefixes_with_space
            if transform is None:
                tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
            else:
                tokens = transform(sent, tokenizer, max_length_per_example)
            prefix = prefixes[(int(label))]

            if method_type == "channel":
                tokens = prefix + tokens
            elif method_type == "direct":
                tokens = tokens + prefix
            else:
                raise NotImplementedError()

            demonstrations += tokens
    if transform is None:
        for i in range(n_classes):
            # based on template add prefix, make sure the input_ids after prefix is same but prefix is not the same
            for j in range(i + 1, n_classes):
                assert prefixes[i][:idx] == prefixes[j][:idx]
                assert prefixes[i][idx] != prefixes[j][idx]
    input_tensors = []

    for i in range(n_classes):
        if transform is None:
            prefix = prefixes[i].copy()
            if method_type == "channel":
                if use_demonstrations:
                    prefix = demonstrations.copy() + prefix
                tensor = prepro_sentence_pair([prefix], test_inputs, max_length,
                                              bos_token_id, eos_token_id,
                                              allow_truncation=use_demonstrations)
            elif method_type == "direct":
                if use_demonstrations:
                    prompt = [demonstrations.copy() + test_input + prefix[:idx] for test_input in test_inputs]
                else:
                    prompt = [test_input + prefix[:idx] for test_input in test_inputs]
                tensor = prepro_sentence_pair(prompt,
                                              [prefix[idx:]], max_length,
                                              bos_token_id, eos_token_id,
                                              allow_truncation=use_demonstrations)
            else:
                raise NotImplementedError()
        else:
            input_ids, attention_mask, token_type_ids = [], [], []
            for input_, output_ in test_inputs:
                encoded = prepro_sentence_pair_single(
                    input_[i], output_[i], max_length,
                    bos_token_id,
                    None if is_generation else eos_token_id,
                    allow_truncation=False)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
            tensor = dict(input_ids=torch.LongTensor(input_ids),
                          attention_mask=torch.LongTensor(attention_mask),
                          token_type_ids=torch.LongTensor(token_type_ids))

        input_tensors.append(tensor)

    return input_tensors


def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}


def prepro_sentence_pair_single(ids1, ids2, max_length, bos_token_id, eos_token_id, negate=False,
                                allow_truncation=False):
    assert not negate

    if bos_token_id is not None:
        ids1 = [bos_token_id] + ids1
    if eos_token_id is not None:
        ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1) + len(ids2) > max_length:
        ids1 = ids1[len(ids1) + len(ids2) - max_length:]  # len = max_length-len(ids2)
        assert len(ids1) + len(ids2) == max_length

    n_mask = max_length - len(ids1) - len(ids2)
    assert n_mask >= 0, (max_length, len(ids1), len(ids2))
    input_ids = ids1 + ids2 + [0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1 + ids2] + [0 for _ in range(n_mask)]
    if negate:
        token_type_ids = [0 for _ in ids1] + [-1 for _ in ids2] + [0 for _ in range(n_mask)]
    else:
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids

def prepare_data_for_parallel(tokenizer, train_data, test_data,
                              max_length, max_length_per_example,
                              method_type, n_classes,
                              test_inputs, prefixes, idx, prefixes_with_space,
                              bos_token_id, eos_token_id):

    # get len(train_data) number of demonstrations

    assert train_data is not None
    demonstrations_list = []

    np.random.shuffle(train_data)

    for sent, label in train_data:
        tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
        prefix = prefixes[(int(label))]
        if method_type=="channel":
            tokens = prefix + tokens
        elif method_type=="direct":
            tokens = tokens + prefix
        else:
            raise NotImplementedError()

        demonstrations_list.append(tokens)

    # check if idx is set well
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            assert prefixes[i][:idx]==prefixes[j][:idx]
            assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):

        if method_type=="channel":
            prefix = prefixes_with_space[i].copy()
            prompt = [demonstrations + prefix
                      for demonstrations in demonstrations_list]
            tensor = prepro_sentence_pair(
                prompt, test_inputs, max_length,
                bos_token_id, eos_token_id,
                allow_truncation=True)

        elif method_type=="direct":
            prefix = prefixes[i].copy()
            prompt = [demonstrations.copy() + test_input + prefix[:idx]
                      for test_input in test_inputs
                      for demonstrations in demonstrations_list]

            tensor = prepro_sentence_pair(prompt,
                                          [prefix[idx:]], max_length,
                                          bos_token_id, eos_token_id,
                                          allow_truncation=True)
        else:
            raise NotImplementedError()

        input_tensors.append(tensor)


    return input_tensors

def main(logger, args):
    args.gpt2 = args.gpt2.replace("gpt2-small", "gpt2")
    tokenizers = GPT2Tokenizer.from_pretrained(args.gpt2)
    model = None

    # train_task default is None
    if args.train_task is None:
        train_task = args.task
    else:
        train_task = args.train_task
        assert args.do_check
    long_datasets = ["ADRD"]
    max_length = 256
    batch_size = int(args.batch_size / 2)

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        # here the demonstrations is to do the zero shot
        assert args.do_zeroshot and not args.do_train
    if args.ensemble:  # TODO why if ensemble means must be demonstrations
        assert args.use_demonstrations

    if args.do_train or args.use_demonstrations:
        assert args.train_seed > 0

    n_templates = 4  # TODO all the templates I used here is 4
    k = int(args.k)
    seed = int(args.seed)

    train_data = load_data(args.data_dir, train_task, k, seed, "train")

    # args.split is define where is the place to use dev
    if args.split is None:
        assert args.do_zeroshot
        dev_data = None
    else:
        dev_data = load_data(args.data_dir, args.task, k, seed, args.split)

    accs = []

    for template_idx in range(n_templates):
        acc = run(logger, args.do_train, not args.do_zeroshot,
                  args.task, train_task,
                  k, seed, args.train_seed,
                  args.out_dir, args.split.tokenizer, model, train_data,
                  dev_data, batch_size, max_length, args.gpt2,
                  template_idx, args.method, args.lr, args.warmup_steps,
                  use_demonstrations=args.use_demonstrations,
                  ensemble=args.ensemble,
                  is_null=args.split is None,
                  prompt_tune=args.prompt_tune,
                  head_tune=args.prompt_tune,
                  transform_tune=args.transform_tune,
                  do_check=args.do_check,
                  n_prefix=args.n_prefix)
        accs.append(acc)
    if args.split is not None:
        logger.info("Accuracy = %.1f(Avg) / %.1f(Worst)" % (100 * np.mean(accs), 100 * np.min(accs)))


def run(logger: object, do_train: object, do_zeroshot: object, task: object, train_task: object, k: object,
        seed: object,
        train_seed: object,
        out_dir: object, split: object, tokenizer: object, model: object,
        train_data: object, dev_data: object,
        batch_size: object, max_length: object, gpt2: object, template_idx: object, method_type: object,
        learning_rate: object, warmup_steps: object,
        use_demonstrations: object = False,
        use_calibration: object = False,
        ensemble: object = False,
        is_null: object = False,
        prompt_tune: object = False,
        head_tune: object = False,
        transform_tune: object = False,
        do_check: object = False, n_prefix: object = 20) -> object:
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)  # Sets the seed for generating random numbers. Returns a torch.Generator object.

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

    if head_tune or transform_tune:
        assert method_type == "direct"
    n_classes = N_LABELS_DICT.get(task, None)
    templates = get_prompt(task, template_idx)

    n_classes_train = N_LABELS_DICT.get(train_task, None)
    templates_train = get_prompt(train_task, template_idx)

    if task in ["ADRD"] and train_task in ["ADRD"]:
        templates = [t.replace(".", " .") for t in templates]

    max_length_per_example = max_length

    if use_demonstrations and not ensemble:
        assert do_zeroshot and not do_train  # means here is
        mem = batch_size * max_length
        max_length = 1024

        max_length = min(max_length, 1024)
        batch_size = int(mem / max_length)

    if do_zeroshot:
        # id is zeroshot using the type based on the template index
        cache_path = [get_paths(out_dir, gpt2, method_type, task, do_zeroshot,
                                k, seed, train_seed, split, template_idx,
                                use_demonstrations=use_demonstrations,
                                ensemble=ensemble)]
        checkpoints = [None]
    else:
        out_dir = get_paths(out_dir, gpt2, method_type, train_task, do_zeroshot,
                            k, seed, train_seed, split, template_idx,
                            batch_size, learning_rate, warmup_steps,
                            use_demonstrations=use_demonstrations,
                            ensemble=ensemble,
                            prompt_tune=prompt_tune,
                            head_tune=head_tune,
                            transform_tune=transform_tune,
                            n_prefix=n_prefix)

        k = int(k)
        eval_period = 100
        num_training_steps = 400

        cache_paths = [os.path.join(out_dir, "{}cache-{}-{}.pkl".format(
            task + "_" if train_task != task else "",
            split, step))
                       for step in
                       range(eval_period, num_training_steps + eval_period, eval_period)]  # range(start, stop, step)
        checkpoints = [os.path.join(out_dir, "model-{}.pt".format(step))
                       for step in range(eval_period)]
    mapping = None

    if do_train and (head_tune or not do_check):
        inputs = prepare_data(tokenizer, None, train_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes_train,
            templates=templates_train,
            method_type=method_type,
            is_training=True,
            ensemble=ensemble)

        logger.info(out_dir)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if not do_check:

            model = GPT2LMHeadModel.from_pretrained(gpt2)

            if prompt_tune: # TODO here is just do the prompt tuning
                for param in model.parameters():  # get all parameters in model
                    param.requires_grad = False  # TODO freeze model weights

                set_extra_embeddings(model, n_prefix)
                inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

            elif head_tune:
                mapping, inputs = reassign_output_tokens(inputs, for_labels=True)
                logger.info("Creare mapping with {} vocabs".format(len(mapping)))
                set_separate_lm_head(model, mapping)
                for param in model.parameters():
                    param.requires_grad = False
                for params in model.lm_head.transform.parameters():
                    params.requires_grad = True

            elif transform_tune:
                set_transformed_lm_head(model)
                for param in model.parameters():
                    param.requires_grad = False # this function is to frozen the paramters
                for param in model.lm_head.transform.parameters():
                    param.requires_grad = True # make the trans
                    
            model = model.cuda()
            
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            train(logger, model, inputs, batch_size, out_dir,
                  learning_rate=learning_rate,
                  warmup_steps=warmup_steps,
                  eval_period=eval_period,
                  num_training_steps=num_training_steps,
                  prompt_tune=prompt_tune,
                  head_tune=head_tune,
                  transform_tune=transform_tune
                  )
    input_tensors = prepare_data(
            tokenizer, train_data,dev_data,
        max_length=max_length,
        max_length_per_example=max_length_per_example,
        n_classes=n_classes,
        templates=templates,
        method_type=method_type,
        use_demonstrations=use_demonstrations,
        ensemble=ensemble,
        is_null=is_null
    )
    if prompt_tune:
        input_tensors = prepend_task_tokens(tokenizer, input_tensors, n_prefix)

    if head_tune:
        if task != train_task:
            if task in ["sst-5", "yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
                input_tensors = [input_tensors[0], input_tensors[-1]]
                if head_tune:
                    label_counter = {'0': '0', '4': '1'}
                    dev_data = [(x, label_counter.get(y, '-1')) for x, y in dev_data]
            elif task in ["SST-2", "mr"] and train_task in ["SST-2", "mr", "sst-5"]:
                pass
            else:
                raise NotImplementedError()
        if mapping is None:
            mapping, inputs = reassign_output_tokens(inputs, for_labels=head_tune)

        train_labels = set([label for _, label in train_data])
        if len(train_labels)!=n_classes:
            train_labels = sorted(train_labels)
            input_tensors = [input_tensors[int(l)] for l in train_labels]
            dev_data = [(sent, str(train_labels.index(l)) if l in train_labels else -1)
                        for sent, l in dev_data]
            _, input_tensors = reassign_output_tokens(input_tensors, for_labels=head_tune,
                                                      mapping={v:k for k, v in mapping.items()})
            logger.info(mapping)
            logger.info("Checked that train mapping and test mapping are identical")

    # for debugging ...
    logger.info("Checking the first example...")
    input_ids = input_tensors[0]["input_ids"][0].numpy().tolist()
    token_type_ids = input_tensors[0]["token_type_ids"][0].numpy().tolist()
    logger.info("Input:")
    logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
    logger.info("Output:")
    logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
