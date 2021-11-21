import pdb
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_text(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    print("=============================================================")
    print("Actual text: \n",filedata)
    print("=============================================================")
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def similarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]
    all_words = list(set(sentence1+sentence2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sentence1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sentence2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1-cosine_distance(vector1, vector2)

def similarity_matrix(sentences, stop_words):
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            sim_matrix[idx1][idx2] = similarity(sentences[idx1], sentences[idx2], stop_words)
    return sim_matrix

def text_summary(file_name, top_n=5):
    # pdb.set_trace()
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_text(file_name)
    sentence_sim_matrix = similarity_matrix(sentences, stop_words)
    sentence_sim_graph = nx.from_numpy_array(sentence_sim_matrix)
    scores = nx.pagerank(sentence_sim_graph)
    ranked_sentence = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summarized text: \.n", ".".join(summarize_text))


text_summary("test.txt")