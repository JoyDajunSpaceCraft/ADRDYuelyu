import math
import os
import random
import string
import pandas as pd
import nltk
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# Import functions for NLP
# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN
#Import model for similarity calculation
import spacy
# nlp = spacy.load("en_core_web_sm")
import en_core_sci_sm
nlp = en_core_sci_sm.load()
#Use spacy to find similarities between tags

tag_list =  ["agingParents", "alzheimers", "alzheimersGroup", "dementia"]
#3871 contains nan so delete it
# del tag_list[3871]

#Get rid of hyphens and turn the split words into an extra tag
corpus = ' '.join(list(tag_list)).replace('-',' ')
words = corpus.split()
corpus = " ".join(sorted(set(words), key=words.index))

#Apply the model on our dataset of tags
tokens = nlp(corpus)

#Convert tags into vectors for our clustering model
word_vectors = []
for i in tokens:
  word_vectors.append(i.vector)
word_vectors = np.array(word_vectors)
#Use cosine because spacy uses cosine. min_samples = 2 because a cluster should have atleast 2 similar words
dbscan = DBSCAN(metric='cosine', eps=0.3, min_samples=2).fit(word_vectors)



# Fit model
random.seed(30)

base_path = "/Users/yuelyu/PycharmProjects/ADRD/collect_Reddit"


def getall():

    csv_name = ["AgingParents", "Alzheimers", "AlzheimersGroup", "Dementia"]

    text_list = []

    for i in csv_name:
        sub_path = os.path.join(base_path, i + ".csv")
        df = pd.read_csv(sub_path)
        for i in list(df.selftext):
            if type(i) != str and math.isnan(i):
                continue
            else:
                text_list.append(i)
    df = pd.DataFrame({"selfbody":text_list})
    df.to_csv("temp.csv", index=False)
    return df


# Functions for NLP
def lowercase(input):
    """
    Returns lowercase text
    """
    return input.lower()


def remove_punctuation(input):
    """
    Returns text without punctuation
    """
    return input.translate(str.maketrans('', '', string.punctuation))


def remove_whitespaces(input):
    """
    Returns text without extra whitespaces
    """
    return " ".join(input.split())


def remove_html_tags(input):
    """
    Returns text without HTML tags
    """
    soup = BeautifulSoup(input, "html.parser")
    stripped_input = soup.get_text(separator=" ")
    return stripped_input


def tokenize(input):
    """
    Returns tokenized version of text
    """
    return word_tokenize(input)


def remove_stop_words(input):
    """
    Returns text without stop words
    """
    input = word_tokenize(input)
    return [word for word in input if word not in stopwords.words('english')]


def lemmatize(input):
    """
    Lemmatizes input using NLTK's WordNetLemmatizer
    """
    lemmatizer = WordNetLemmatizer()
    input_str = word_tokenize(input)
    new_words = []
    for word in input_str:
        new_words.append(lemmatizer.lemmatize(word))
    return ' '.join(new_words)


def nlp_pipeline(input):
    """
    Function that calls all other functions together to perform NLP on a given text
    """
    return lemmatize(
        ' '.join(remove_stop_words(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(input)))))))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Turn tags into a set for faster checking of whether a tag exists or not
unique_tags = ("agingParents", "alzheimers", "alzheimersGroup", "dementia")


def find_topics(question_body):
    """
    Function that takes a question as an input, and finds the two most important topics/tags
    If the found topics exist in the already existing database of tags, we add these tags
    to the professional who answered the question
    """
    try:
        text = nlp_pipeline(question_body)
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform([text])
        # One topic that has an avg of two words because most questions had 1/2 tags
        number_topics = 1
        number_words = 2
        # Create and fit the LDA model
        lda = LDA(n_components=number_topics, n_jobs=-1)
        lda.fit(count_data)

        words = count_vectorizer.get_feature_names()

        # Get topics from model. They are represented as a list e.g. ['military','army']
        topics = [[words[i] for i in topic.argsort()[:-number_words - 1:-1]] for (topic_idx, topic) in
                  enumerate(lda.components_)]
        topics = np.array(topics).ravel()

        # Only use topics for which a tag already exists
        existing_topics = set.intersection(set(topics), unique_tags)

    # A few question bodies don't work with LDA so this exception just prints them out and ignores them
    except:
        print(question_body)
        return (question_body)

    return existing_topics


def clustering_model():
    def __init__(self):
        pass

    def dbscan_predict(self, model, X):

        nr_samples = X.shape[0]

        y_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = model.components_ - X[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < model.eps:
                y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

        return y_new

if __name__ == "__main__":
    # df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask10/temp.csv")
    # new_tags= df["selfbody"].apply(find_topics).values
    print("&"*20)
    # print("new_tags",new_tags)
    # c = clustering_model()
    # c.dbscan_predict(dbscan, )

