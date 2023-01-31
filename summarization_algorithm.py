import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

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