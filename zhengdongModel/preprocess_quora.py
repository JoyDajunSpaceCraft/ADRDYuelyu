import json
import spacy
import os
from bert_embedding import BertEmbedding
import xgboost as xgb
import numpy as np


from bert_sentence_classifier.test_bert_classifier import load_or_create_bert_embedding_for_sentence, \
    get_or_create_sentence_embedding, calculate_embedding, create_sent_offset

nlp = spacy.load("en_core_web_sm",disable=["ner"])


def create_questions(root_folder:str):
    sentences = []
    for file in os.listdir(root_folder):
        if file.endswith(".json"):
            data = json.load(open(os.path.join(root_folder,file)))
            for item in data:
                question_text = item["question"]
                sentences.append(question_text)
    return sentences

def create_answers(root_folder:str):
    answers = []
    for file in os.listdir(root_folder):
        if file.endswith(".json"):
            data = json.load(open(os.path.join(root_folder,file)))
            for item in data:
                question_text = item["answer"]
                doc = nlp(question_text)
                for sent in doc.sents:
                    answers.append(str(sent))
    return answers

def train_xgb_model(model_path:str,sentence_embedding,labels):
    train = np.zeros((len(sentence_embedding),len(sentence_embedding[0])))
    for i,data in enumerate(sentence_embedding):
        train[i] = data
    dtrain = xgb.DMatrix(train, label=labels)
    param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softmax'}
    param["num_class"] = 2
    param['nthread'] = 8
    param['eval_metric'] = 'auc'
    num_round = 700
    bst_model = xgb.train(param, dtrain, num_round)
    bst_model.save_model(model_path)
    return bst_model

def predict_on_alzheimer_data(model,embedding_method:str = "CLS"):
    import pickle
    sentence_embedding_file_path = "/Users/zhendongwang/Documents/projects/is/alzheimer/bert_sentence_classifier/data/bert-old-reddit-sent-embedding.pkl"
    sentence_docId_map_file_path = "/Users/zhendongwang/Documents/projects/is/alzheimer/bert_sentence_classifier/data/train/sentId_docId_map.pkl"
    sentId_docId_map =  pickle.load(open(sentence_docId_map_file_path,"rb"))


    reddit_embedding =  pickle.load(open(sentence_embedding_file_path,"rb"))
    sentId_offset_map,sentId_sentlen_map = create_sent_offset(sentId_docId_map,reddit_embedding)

    reddit_sentence_embedding = []
    for sent_embedding in reddit_embedding:
        tokens = sent_embedding[0]
        embedding = sent_embedding[1]
        if len(tokens) == 0:
            reddit_sentence_embedding.append(None)
        else:
            reddit_sentence_embedding.append(calculate_embedding(tokens,embedding,embedding_method))

    test = np.zeros((len(reddit_sentence_embedding),len(reddit_sentence_embedding[0])))
    for i,embedding in enumerate(reddit_sentence_embedding):
        test[i] = embedding
    dtest = xgb.DMatrix(test)

    prediction = model.predict(dtest)
    doc_sentences_to_highlight = {}

    for i,is_question in enumerate(prediction):
        if is_question == 1:
            doc_id = sentId_docId_map[i]
            if doc_id not in doc_sentences_to_highlight:
                doc_sentences_to_highlight[doc_id] = []
            doc_sentences_to_highlight[doc_id].append({
                "id":i,
                "start":sentId_offset_map[i],
                "end":sentId_offset_map[i] + sentId_sentlen_map[i],
            })
    return doc_sentences_to_highlight

def dump_hiw_sentence(doc_sentences_to_highlight,annotation_root:str):
    for file in os.listdir(annotation_root):
        if file.endswith(".json"):
            json_file_path = os.path.join(annotation_root,file)
            data = json.load(open(json_file_path))
            doc_id = int(file.replace(".json",""))
            if doc_id in doc_sentences_to_highlight:
                data["highlight_sent_hiw"] = doc_sentences_to_highlight[doc_id]
            json.dump(data,open(json_file_path,"w"))





if __name__ == '__main__':
    question_root = "/Users/zhendongwang/Documents/projects/is/alzheimer/quora/data/questions"
    answer_root = "/Users/zhendongwang/Documents/projects/is/alzheimer/quora/data/answer"
    annotation_root = "/Users/zhendongwang/Documents/projects/is/alzheimer/bert_sentence_classifier/data/bert_annotation/users/bo_xie/annotations"


    token_embedding_file_path = "/Users/zhendongwang/Documents/projects/is/alzheimer/quora/data/embedding/all_sentence_embedding.pkl"
    sentence_embedding_file_path = "/Users/zhendongwang/Documents/projects/is/alzheimer/quora/data/embedding/sentence_embedding.pkl"

    model_path = "/Users/zhendongwang/Documents/projects/is/alzheimer/quora/data/model/cls_xgboost.bin"


    questions = create_questions(question_root)
    answers = create_answers(answer_root)
    all_sentences = questions + answers
    labels = [1] * len(questions) + [0] * len(answers)

    bert_embedding = BertEmbedding(dataset_name="book_corpus_wiki_en_cased")
    token_embedding = load_or_create_bert_embedding_for_sentence(token_embedding_file_path, all_sentences, bert_embedding)
    sentence_embedding = get_or_create_sentence_embedding(sentence_embedding_file_path,token_embedding,"CLS")

    model = train_xgb_model(model_path,sentence_embedding,labels)
    doc_sentences_to_highlight = predict_on_alzheimer_data(model,"CLS")
    dump_hiw_sentence(doc_sentences_to_highlight,annotation_root)


