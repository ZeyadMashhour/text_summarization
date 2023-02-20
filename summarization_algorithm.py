"""Imports"""

import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import numpy as np
from numpy import dot
from numpy.linalg import norm

"""Pre Processing"""
def tokenization(text):
    sentences = sent_tokenize(text)
    # count the number of sentences in the text 
    total_sentences = len(sentences)
    #print("Total number of sentences:", total_sentences)
    return sentences

def remove_stop_words(sentences):
    # define a set of stop words 
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        # tokenize each sentence into words
        word_tokens = word_tokenize(sentence)
        # remove all stop words from each sentence
        filtered_words = [word for word in word_tokens if not word in stop_words]
        # join all filtered words back into a single sentence
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


"""Algorithms"""

#textMacthingAlgorithm
def text_matching_algorithm(filtered_sentences, sentences,size = 5):
    
    word_frequencies = {}  # Create an empty dictionary to store the word frequencies
    for sentence in filtered_sentences:  # Loop through each sentence in the text
        words = nltk.word_tokenize(sentence)  # Tokenize each sentence into words
        for word in words:  # Loop through each word in the sentence
            if word not in word_frequencies.keys():  # Check if the word is already in the dictionary
                word_frequencies[word] = 1  # If not, set its count to 1
            else:
                word_frequencies[word] += 1  # If yes, increment its count by 1
    sentence_map = {} # Create an empty dictionary to store the sentence scores
    sentence_scores = []
    sent_index = 0
    for sentence in sentences:  # Loop through each sentence in the text again
        words = nltk.word_tokenize(sentence)  # Tokenize each sentence into words again
        score = 0  # Initialize a score variable to 0
        for word in words:  # Loop through each word in the sentence
            if word in word_frequencies.keys():  # Check if the current word is present in our dictionary of word frequencies
                score += word_frequencies[word]  # If yes, add its frequency to our score variable
        sentence_map[sent_index] = [sentence]
        sentence_scores.append(score)
        sent_index += 1
        
    # scores = sorted(sentence_map.keys())
    # scores = scores[len(scores)-summary_size:]
    maxScore = max(sentence_scores)
    for i in range(len(sentence_scores)):
        sentence_map[i].append(sentence_scores[i] / maxScore)

        
    #return sentence_map
    summary = ""  # Create an empty list to store the summary sentences
    sorted_ix = np.argsort(sentence_scores)[::-1]
    for i in sorted_ix[:size]:
        summary+=sentences[i]
    return summary



#Luhn algorithm
def luhn_algorithm(filtered_sentences, sentences,size = 5):
    # f = open(file_name, "r")
    # text = ""
    # for line in f:
    #     text += line.replace('!','.').replace('?','.').replace('\n',' ')
    # f.close()
    
    # # Split the text into sentences
    # sentences = text.split('.')
    # Initialize a list to store the sentence scores
    sentence_scores = []
    sentence_map = {}
    sent_index = 0
    # Iterate through each sentence
    for sentence in filtered_sentences:
        # Split the sentence into words
        words = sentence.split()
        # Initialize a score for the sentence
        score = 0
        # Iterate through each word
        for word in words:
            # Calculate the score for the word
            score += len(word)
        # Add the score to the sentence scores list
        sentence_scores.append(score)
        sentence_map[sent_index] = [sentences[sent_index]]
        sent_index += 1

    maxScore = max(sentence_scores)
    for i in range(len(sentence_scores)):
        sentence_map[i].append(sentence_scores[i] / maxScore)
    #return sentence_map

    summary = ""  # Create an empty list to store the summary sentences
    sorted_ix = np.argsort(sentence_scores)[::-1]
    for i in sorted_ix[:size]:
        summary+=sentences[i]
    return summary


def create_tf_idf(filtered_sentences):
    tfidfconverter = TfidfVectorizer()
    X = tfidfconverter.fit_transform(filtered_sentences).toarray()
    return X

def lsa_algorithm(X):
    svdmodel = TruncatedSVD(n_components=2)
    svdmodel.fit(X)
    result = svdmodel.transform(X)
    return result

def lsa_summarization(filtered_sentences, sentences,size = 5):
    # file = open(file_name, "r")
    # text = file.read()
    # file.close()
    
    # sentences = tokenization(text)
    # filtered_sentences = remove_stop_words(sentences)
    X = create_tf_idf(filtered_sentences)
    result = lsa_algorithm(X)
    sentence_scores = result[:,1]
    sentence_map = {}
    normalized = (sentence_scores-min(sentence_scores))/(max(sentence_scores)-min(sentence_scores))
    # sent_index = 0
    # summarize our text by selecting only those sentences with higher sentence_scores
    for i in range (len (normalized)):
        sentence_map[i] = [sentences[i], normalized[i]]
    #return sentence_map

    summary = ""  # Create an empty list to store the summary sentences
    sorted_ix = np.argsort(sentence_scores)[::-1]
    for i in sorted_ix[:size]:
        summary+=sentences[i]
    return summary


#########################
#LexRank_algorithm
########

def LexRank_algorithm(filtered_sentences,sentences,size=5,threshold = 0.095):
     #creating tf_idf
    tfidfconverter = TfidfVectorizer()
    tf_idf = tfidfconverter.fit_transform(filtered_sentences).toarray()
    
    #sentences length
    sent_length = []
    for i in range(len(tf_idf)):
        tf_idf_length = 0
        for sent_tf_idf in tf_idf[i]:
            tf_idf_length += math.sqrt(sent_tf_idf)**2
        sent_length.append(tf_idf_length)
        
    #normalized tf_idf
    normalized_tf_idf = []
    # for row in range(len(tf_idf)): 
    #     for col in range(len(tf_idf[row])):
    #         if math.isclose(tf_idf[row,col],0):
    #             tf_idf[row,col] = 0
    #         else:
    #             tf_idf[row,col] = tf_idf[row,col]/sent_length[row]
    new_tf_idf = np.zeros(tf_idf.shape)
    
    for row in range(len(tf_idf)): 
        for col in range(len(tf_idf[row])):
            new_tf_idf[row,col] = tf_idf[row,col]/sent_length[row] 
    normalized_tf_idf = new_tf_idf
            
    length = len(normalized_tf_idf)
    similarity_matrix = np.zeros([length] * 2)
    
    for i in range(length):
        for j in range(i, length):
            similarity = cosine_similarity(normalized_tf_idf[i],normalized_tf_idf[j],i,j)

            if similarity:
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity    
    
    def get_summary(sentences,similarity_matrix,threshold,summary_size=1):

        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = rank_sentences(sentences,similarity_matrix,threshold)

        sorted_ix = np.argsort(lex_scores)[::-1]

        summary_index=[]
        for i in sorted_ix[:summary_size]:
            summary_index.append(i)
        #print(summary_index)
        return lex_scores,summary_index

    scores , summary_index = get_summary(sentences,similarity_matrix,threshold,size)
    summary = ""
    for i in summary_index:
        summary += sentences[i]       
    return summary
###############################

def connected_nodes(matrix):
    _, labels = connected_components(matrix)
    z = csr_matrix(matrix)
    groups = []
    for tag in np.unique(labels):
        #returns an array with elements from x where condition is True, and elements from y elsewhere.
        group = np.where(labels == tag)[0]
        groups.append(group)
    return groups


#cosine similarity
def cosine_similarity(list_1, list_2,i,j):
        if i == j :
            return 1
        dot = np.dot(list_1, list_2)
        if math.isclose(dot, 0):
            return 0
        norm = (np.linalg.norm(list_1) * np.linalg.norm(list_2))
        cos_sim = dot / norm
        return cos_sim
    
    
def stationary_distribution(transition_matrix,normalized=True):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'transition_matrix\' should be square')

    distribution = np.zeros(n_1)
    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix)
        distribution[group] = eigenvector
    if normalized:
        distribution /= n_1
    return distribution

# transition_matrix is a Stochastic matrix, a square matrix used to describe the transitions of a Markov chain.
def _power_method(transition_matrix):
    sentences_count = len(transition_matrix)
    eigenvector = np.ones(sentences_count)
    if len(eigenvector) == 1:
        return eigenvector
    transposed_matrix = transition_matrix.T
    lambda_val = 1.0

    while np.allclose(lambda_val, eigenvector):
        eigenvector_next = np.dot(transposed_matrix, eigenvector)
        lambda_val = np.linalg.norm(np.subtract(eigenvector_next, eigenvector))
        eigenvector = eigenvector_next
    return eigenvector


def create_markov_matrix(weights_matrix):
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'weights_matrix\' should be square')

    row_sum = weights_matrix.sum(axis=1, keepdims=True)

    return weights_matrix / row_sum

def create_markov_matrix_discrete(weights_matrix, threshold):
    discrete_weights_matrix = weights_matrix#np.zeros(weights_matrix.shape)
    #print(discrete_weights_matrix)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1
    #print(discrete_weights_matrix)

    return create_markov_matrix(discrete_weights_matrix)

def degree_centrality_scores(similarity_matrix,threshold=None,increase_power=True):
    if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
        raise ValueError(
            '\'threshold\' should be a floating-point number '
            'from the interval [0, 1) or None')

    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)

    else:
        markov_matrix = create_markov_matrix_discrete(similarity_matrix,threshold)

    scores = stationary_distribution(markov_matrix,normalized=True)
    return scores

def rank_sentences(sentences,similarity_matrix,threshold=0.03):  
    scores = degree_centrality_scores(similarity_matrix,threshold)
    return scores





def summarize_with(list_of_articles,list_of_filtered_articles ,summary_algorithm,size = 2):
    rows = len(list_of_articles)
    
    summarized_text = []
    for row in tqdm(range(rows)):
        sentences=list_of_articles[row]
        filtered_sentences = list_of_filtered_articles[row]
                                  #(filtered_sentences,sentence)
        summary = summary_algorithm(filtered_sentences,sentences,size)
        summarized_text.append(summary)
    summary_df = pd.DataFrame (summarized_text, columns = [f'{summary_algorithm.__name__} summary'])
    return summary_df

