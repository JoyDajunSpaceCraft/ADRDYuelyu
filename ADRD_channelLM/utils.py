import torch


def prepend_task_tokens(tokenizer, inputs, n_prefix):
    # for example n_prefix is 3 the zfill(2) means 00 01 02
    task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(n_prefix)]

    tokenizer.add_tokens(task_tokens)  # token inputs id is add as sentences
    task_token_ids = tokenizer(" ".join(task_tokens), return_tensors="pt")["input_ids"]
    assert task_token_ids.shape[-1] == n_prefix  # makesure the n_prefix add in the last

    def convert(inputs):
        n_train = inputs["input_ids"].shape[0]  # get how much sentences in the training set

        new_input_ids = torch.cat([
            task_token_ids.repeat(n_train, 1),  # make the new prefix data repeat itself once
            inputs["input_ids"][:, 1:]  # why begin from 1 the begin [CLS]
        ], 1)  # here the last 1 means cat in col

        inputs = dict(
            input_ids=new_input_ids,
            attention_mask=torch.cat([
                torch.ones((n_train, n_prefix - 1), dtype=torch.long),
                inputs["attentions_mask"]], 1),
            token_type_ids=torch.cat([
                torch.zeors((n_train, n_prefix - 1), dtype=torch.long),
                inputs["token_type_ids"]], 1),
            labels=torch.cat([
                torch.zeros((n_train, n_prefix - 1), dtype=torch.long),
                inputs["input_ids"]
            ], 1))
        return inputs
    if type(inputs) == list:
        return [convert(_inputs) for _inputs in inputs]
    return convert(inputs)

def reassign_output_tokens(inputs, for_labels=True, mapping=None):
    '''
    if for_labels=True, keep input_ids and convert labels
    otherwise, keep labels and convert input_ids
    '''
    # the input_ids["token_type_ids"] should [0,1,1,1,1] which is turn as True and False
    def get_unique_tokens(inputs):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach().numpy()
        unique_tokens = set()
        for _input_ids, _token_type_ids in zip(input_ids, token_type_ids):
            unique_tokens|=set([_id for _id, _token_id in zip(_input_ids, _token_type_ids) if _token_id == int(for_labels)])
        return unique_tokens

    def convert_set_to_mapping(unique_tokens):
        unique_tokens = sorted(unique_tokens)
        return {token:new_token for new_token, token in enumerate(unique_tokens)}

    def apply_mapping(input, mapping):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach.numpy().tolist()
        converted_input_ids = []
        for _input_ids, token_type_ids in zip(input_ids, token_type_ids):
            converted_input_ids.append([])
            for _id, _token_id in zip(input_ids, token_type_ids):
                if _token_id == int(for_labels):
                    converted_input_ids[-1].append(mapping[_id])
                else:
                    converted_input_ids[-1].append(0)
        converted_input_ids = torch.LongTensor(converted_input_ids)

        if for_labels:
            return dict(input_ids=input["input_ids"],
                        attention_mask=input["attention_mask"],
                        token_type_ids=input["token_type_ids"],
                        labels=converted_input_ids)
        return dict(input_ids=input["input_ids"],
                        attention_mask=input["attention_mask"],
                        token_type_ids=input["token_type_ids"],
                        labels=input["input_ids"])

    if type(inputs) == list:
        if mapping is None:
            unique_tokens = set()
            for _inputs in inputs:
                unique_tokens |= get_unique_tokens(_inputs)
            mapping = convert_set_to_mapping(unique_tokens)
        rev_mapping = {v: k for k, v in mapping.items()}
        return rev_mapping, [apply_mapping(_inputs, mapping) for _inputs in inputs]

    assert mapping is None
    mapping = convert_set_to_mapping(get_unique_tokens(inputs))
    rev_mapping = {v: k for k, v in mapping.items()}
    return rev_mapping, apply_mapping(inputs, mapping)











