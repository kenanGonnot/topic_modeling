import io
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # , RobustScaler, robust_scale, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import spacy

import nltk

nltk.download('stopwords')
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
# %% md
### Load data

from yellowbrick.cluster import KElbowVisualizer
import time

np.random.seed(5)


# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def gen_words(texts):
    final = []
    for text in texts:
        new = yield (simple_preprocess(str(text), deacc=True))  # deacc True remove punctuations
        final.append(new)
    return final


def preprocess_input_text_(text):
    # Remove new line characters
    text = [re.sub('\s+', ' ', sent) for sent in text]
    # Remove distracting single quotes
    text = [re.sub("\'", "", sent) for sent in text]
    text = "".join(text)
    # text = str(text)
    text = text.split(".")
    text = [sentence for sentence in text if not int(len(sentence)) < 6]
    return text


def pca_dimension_reduction(embeddings):
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(embeddings)  # rescale embeddings
    # find the optimal number of components for PCA
    pca = PCA(n_components=0.95)
    result = pca.fit(data_rescaled)
    y = np.cumsum(result.explained_variance_ratio_)
    n_components = [index for index, value in enumerate(y) if value > 0.95][0]
    # apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_rescaled)
    return pca_result


def find_optimal_k_cluster(pca_result):
    # find the optimal number of clusters
    ks = range(2, 10)
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=ks)
    visualizer.fit(pca_result)
    n_clusters_optimal = visualizer.elbow_value_
    return n_clusters_optimal


def find_nearest_topic_clusters(kmeans_model, pca_result, centers, n=5):
    top_nearest_indices_by_clusters = []
    for cluster in range(len(centers)):
        ind = np.argsort(kmeans_model.transform(pca_result)[:, cluster])[:n]
        top_nearest_indices_by_clusters.append(ind)
    return top_nearest_indices_by_clusters


def create_dataframe_from_top_nearest_indices(top_nearest_indices_by_clusters, text):
    all_documents = []
    for cluster in top_nearest_indices_by_clusters:
        document = []
        for index in cluster:
            if int(len(text[index]) <= int(4)): continue
            text[index] = text[index] + ". "
            document.append(text[index])
        all_documents.append(document)
    courses = {
        'documents': all_documents
    }
    documents = pd.DataFrame(courses)
    return documents


def preprocess_documents_(documents):
    documents = documents['documents']
    return documents
    # data_words = list(gen_words(documents))
    # data_words_nostops = remove_stopwords(data_words)
    # data_words_bigrams = make_bigrams(data_words_nostops)
    # data_lemmatized = lemmatization(data_words_bigrams)


def prepare_bigram_trigram(data_words):
    '''
    Bigrams are 2 words frequently occuring together in docuent.
    Trigrams are 3 words frequently occuring.
    Many other techniques are explained in part-1 of the blog which are important in NLP pipline, it would be worth your while going through that blog.
    The 2 arguments for Phrases are min_count and threshold. The higher the values of these parameters , the harder its for a word to be combined to bigram.

    '''

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    # higher threshold fewer phrases
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return bigram_mod, trigram_mod


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))
    return texts_out


def preprocess_documents_tf_idf_(data_words, bigram_mod):
    print("data_words: ", data_words)
    data_words_nostops = remove_stopwords(data_words)
    print("data_words_nostops: ", data_words_nostops)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    print("data_words_bigrams: ", data_words_bigrams)
    data_lemmatized = lemmatization(data_words_bigrams)
    print("data_lemmatized: ", data_lemmatized)
    return data_lemmatized


def tf_idf(data_lemmatized):
    tfIdfVectorizer = TfidfVectorizer(analyzer='word', stop_words='english', use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(data_lemmatized)
    feature_names = tfIdfVectorizer.get_feature_names()
    return tfIdf, feature_names


def topic_extraction(text, embed):
    print("inside topic extraction")
    print(">>>>>>>TEXTE>>>>>>> , ", text)
    text = preprocess_input_text_(text)
    print(">>>>>>>PREPROCESSED TEXTE>>>>>>> , ", text)
    print("1 - preprocess_input_text_ ----- done ")
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # embeddings = embed(text)
    # embeddings = embed.encode(text)
    # embeddings = embed(text, signature="default", as_dict=True)["elmo"]
    start_embedding = time.time()
    data_list = [embed(doc).vector.reshape(1, -1) for doc in text]
    print("2 - embedding ----- done time: ", time.time() - start_embedding)
    print("type of data_list: ", type(data_list))
    embeddings = np.concatenate(data_list)
    embeddings = np.array(embeddings)
    print(" >>>>>>> EMBEDDING_DATA : ", embeddings)
    print("type of embeddings: ", type(embeddings))
    print("3 - concatenate ----- done ")
    start_pca = time.time()
    pca_result = pca_dimension_reduction(embeddings)
    print(" >>>>>>> PCA_DATA : ", pca_result)
    print("type of pca_result: ", type(pca_result))
    print("4 - pca ----- done time: ", time.time() - start_pca)

    start_kmeans = time.time()
    n_clusters_optimal = find_optimal_k_cluster(pca_result)
    if n_clusters_optimal is None:
        n_clusters_optimal = 2
    print(" >>>>>>> n_clusters_optimal : ", n_clusters_optimal)
    print("5 - find_optimal_k_cluster ----- done time: ", time.time() - start_kmeans)
    start_kmeans = time.time()
    kmeans_model = KMeans(n_clusters=n_clusters_optimal, random_state=0)
    kmeans_model.fit(pca_result)
    print("6 - kmeans_model.fit ----- done time: ", time.time() - start_kmeans)
    centers = np.array(kmeans_model.cluster_centers_)

    label = kmeans_model.fit_predict(pca_result)

    plt.figure(figsize=(15, 15))
    uniq = np.unique(label)
    for i in uniq:
        plt.scatter(pca_result[label == i, 0], pca_result[label == i, 1], label=i)

    for center in centers:
        plt.scatter(center[0], center[1], marker="*", c="r", alpha=1)
    plt.legend()
    plt.show()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # label = kmeans_model.fit_predict(pca_result)
    print("7 - kmeans_model.fit_predict ----- done ")
    top_nearest_indices_by_clusters = find_nearest_topic_clusters(kmeans_model, pca_result, centers, n=5)
    print("type of top_nearest_indices_by_clusters: ", type(top_nearest_indices_by_clusters))
    print("8 - find_nearest_topic_clusters ----- done ")
    documents = create_dataframe_from_top_nearest_indices(top_nearest_indices_by_clusters, text)
    print("9 - create_dataframe_from_top_nearest_indices ----- done ")
    documents = preprocess_documents_(documents)
    print("preprocessed documents ", documents)
    temp = []
    for i in range(len(documents)):
        # temp.append(documents[i])
        temp.append("".join(documents[i]))
    documents = temp
    documents = " ".join(documents)
    # print("documents", documents)
    # print(" type of documents", type(documents))

    data_words = list(gen_words(temp))

    bigram_mod, trigram_mod = prepare_bigram_trigram(data_words)

    data_lemmatized = preprocess_documents_tf_idf_(data_words, bigram_mod)
    tfIdf, feature_names = tf_idf(data_lemmatized)

    all_cluster = []
    for i in range(len(data_lemmatized)):
        probabilities = tfIdf[i].T.todense().A1.tolist()
        print("probabilities shape", len(probabilities))
        print("probabilities", probabilities)
        print("feature name: ", len(feature_names))
        df_tfidf = pd.DataFrame({"TF-IDF": probabilities, "word": feature_names})
        df_tfidf = df_tfidf.sort_values('TF-IDF', ascending=False)
        df_tfidf['cluster'] = i
        all_cluster.append(df_tfidf.head(5))

    result_df = pd.concat(all_cluster)

    return documents, top_nearest_indices_by_clusters, img, data_lemmatized, result_df


if __name__ == '__main__':
    print('qsd')
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # topic_extraction(text)
