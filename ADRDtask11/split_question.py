import os
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from wordcloud import WordCloud
import json
import pickle
import pyLDAvis
import pyLDAvis.gensim_models
from pprint import pprint
from copy import deepcopy

import math
import nltk
from nltk import sent_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd

import spacy
# spacy.load('en_core_web_sm')


class generate_QB:
    def __init__(self):
        self.df = pd.read_csv("all_new_modify_csv.csv")
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(["get","day","make",'from',"know", 'mom',"dad",'re', 'edu', 'use', "father", "time", "year", "years",
                           "want", "go", "help", "one", "really", "would", "much", "many", "could", "also","going","even","take"])
        self.modify_text = self.df.modify_text.values.tolist()
        new_str = []
        for i in self.modify_text:
            if type(i) is str:
                new_str.append(i)
            else:
                new_str.append("")
        self.modify_text =new_str
        self.data_words = list(self.sent_to_words(self.modify_text))
        self.data_words = self.remove_stopwords(self.data_words)


    def build_word_cloud(self):
        # Join the different processed titles together.
        tmp_sent = []
        for i in self.data_words:
            tmp_sent.append(" ".join(i))
        long_string = ','.join(tmp_sent)
        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        # Generate a word cloud
        wordcloud.generate(long_string)
        wordcloud = wordcloud.to_file("wordcloud.png")

    def make_bigrams(self, texts):
        # remove stop words
        # Build the bigram and trigram models
        self.bigram = gensim.models.Phrases(self.data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)

        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self, texts):
        # the trigram first need bigrams
        self.make_bigrams(texts)
        self.trigram = gensim.models.Phrases(self.bigram[self.data_words], threshold=100)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)
        return [self.trigram_mod[doc] for doc in texts]


    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def generate_corpora(self):
        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(self.data_words)
        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'VERB'])
        print(data_lemmatized[:1])
        # Create Dictionary
        self.id2word = corpora.Dictionary(data_lemmatized)
        print(self.id2word)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]
        # View
        print(self.corpus[:1])

    def sent_to_words(self,sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield (simple_preprocess(str(sentence), deacc=True))

    def remove_stopwords(self,texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in self.stop_words] for doc in texts]


    def generate_topics(self, num_topics=5):

        lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=num_topics,
                                               iterations=100)
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())

        doc_lda = lda_model[self.corpus]
        return lda_model.print_topics()
# this function is to clean all of the data in the "Alzheimers", "AgingParents", "Dementia" submissions
# remove the http and lower case and remove \n
def clean():
    csv_file_base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"
    base_list = ["Alzheimers", "AgingParents", "Dementia"]
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, [os.path.join(csv_file_base_path, i + ".csv") for i in base_list]), ignore_index=True)
    post = df.selftext.values.tolist()
    comment = df.comment.values.tolist()
    selfbody = []
    for p, c in zip(post, comment):
        p = str(p)
        c = str(c)
        if p is None and c is None:
            selfbody.append("")
        else:
            self_body = " ".join([p, c])
            if "http" or "https" in self_body:
                self_body = re.sub(r'http\S+', '', self_body)

            selfbody.append(self_body)
    df["modify_text"] = selfbody
    df = df.drop(["selftext", "comment"], axis=1)
    # df['modify_text'] = df['modify_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    df['modify_text'] = df['modify_text'].map(lambda x: x.lower())
    df['modify_text'] = df['modify_text'].map(lambda x: re.sub('\n', " ", x))
    # Print out the first rows of papers
    df.to_csv("all_new_modify_csv.csv", index=False)

def gerneate_predict():
    final_output = []
    # here we get one_qa with format like this:
    # {
    #     "answers": [
    #         {
    #             "answer_start": 28,
    #             "text": "between 2006 and 2008"
    #         }
    #     ],
    #     "id": "00002",
    #     "is_impossible": False,
    #     "question": "When was the series published?"
    # },
    # one_qa = {}

    dailycare_post = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/dailycare_.csv"
    dailycare =  pd.read_csv(dailycare_post).selfbody.to_list()

    qa_content = {}
    id = 0
    for item in dailycare:
        one_qa = {}
        sentences = sent_tokenize(item)

        question = [sentence for sentence in sentences if "?" in sentence]
        if len(question)>0:
            question = question[-1]
        else:
            question = sentences[-1]
        one_qa["question"] = question
        qa_content["qas"] = []

        one_qa["is_impossible"] = False
        one_qa["id"] = str(id)
        id += 1
        answer = {}
        answer["answer_start"], answer["text"] = get_start_index(question, sentences, item)

        one_qa["answers"] = [answer]

        temp_qa = deepcopy(one_qa)
        qa_content["qas"].append(temp_qa)
        qa_content["context"] = item

        final_output.append(qa_content)
        qa_content = {}

    print(final_output)

    with open("predict.json", "w") as wri:
        json.dump(final_output, wri)

def get_start_index(question, sentences, content):

    index = content.index(question)

    question_index = None
    for idx, i in enumerate(sentences):
        if question in i or i in question:
            question_index = idx
            break
    if index + len(question) < len(content):
        text = " ".join(sentences[question_index: question_index + 2])
    else:
        text = " ".join(sentences[question_index - 2: question_index])
    return index, text


def combine_qa_result():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/bertQANew/pred_json_bert.json"
    res = []
    with open(path, "r") as f:
        j = json.load(f)
    for key, value in j.items():
        if value["answer"] is None:
            continue
        else:
            res.append(" ".join([value["question"], value["answer"]]))
    res_df = pd.DataFrame({"selfbody": res})
    res_df.to_csv("dailycare_sentence.csv",index=False)


if __name__ == "__main__":
    # spacy.load('en_core_web_sm')
    # clean()

    # gerneate_predict()
    combine_qa_result()
    # s = generate_QB()
    # s.generate_corpora()
    # topics = []
    # for i in range(5,11):
    #     topics.append(s.generate_topics(i))
    # with open("store_topics.json", "w") as f:
    #     json.dump(topics,f)

    # s.build_word_cloud()


    # df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/test.tsv",sep="\t")
    #
    # pred_txt = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask11/pred.txt"
    # index = []
    # with open(pred_txt, "r") as f:
    #     for idx, i in enumerate(f.readlines()):
    #         if i.split(" ")[-1].split("\n")[0] == "1":
    #             index.append(int(i.split(" ")[0]))
    # print(index)
    # new_df = df.iloc[df.index.isin(index)]
    # new_df.to_csv("dailycare.csv", index=False)
    # print(new_df)










