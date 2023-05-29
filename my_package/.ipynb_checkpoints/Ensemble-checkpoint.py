import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import tqdm as tqdm

def text_matching(filtered_sentences, sentences):
    word_frequencies = {} 
    for sentence in filtered_sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    summary = []
    sentence_map = {}
    sent_scores = []
    sent_index = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        score = 0 
        for word in words:
            if word in word_frequencies.keys():
                score += word_frequencies[word]
        sentence_map[sent_index] = [sentence]
        sent_scores.append(score)
        sent_index += 1
    maxScore = max(sent_scores)    
    sortedScored = list(reversed(sorted(sent_scores)))
    
    for i in range(len(sent_scores)):
        sentence_map[i].append(sent_scores[i] / maxScore)
        sentence_map[i].append(sortedScored.index(sent_scores[i]))
    return sentence_map

def luhn_algorithm(filtered_sentences, sentences):
    summary = []
    sentence_scores = []
    sentence_map = {}
    sent_index = 0
    for sentence in filtered_sentences:
        words = sentence.split()
        score = 0
        for word in words:
            score += len(word)
        sentence_scores.append(score)
        sentence_map[sent_index] = [sentences[sent_index]]
        sent_index += 1
    maxScore = max(sentence_scores)
    sortedScored = list(reversed(sorted(sentence_scores)))
    for i in range(len(sentence_scores)):
        sentence_map[i].append(sentence_scores[i] / maxScore)
        sentence_map[i].append(sortedScored.index(sentence_scores[i]))
    return sentence_map

def create_tf_idf(filtered_sentences):
    tfidfconverter = TfidfVectorizer()
    X = tfidfconverter.fit_transform(filtered_sentences).toarray()
    return X

def lsa_algorithm(X):
    svdmodel = TruncatedSVD(n_components=2)
    svdmodel.fit(X)
    result = svdmodel.transform(X)
    return result

def lsa_summarization(filtered_sentences, sentences):
    X = create_tf_idf(filtered_sentences)
    result = lsa_algorithm(X)
    scores = result[:,1]
    summary = ""
    sentence_map = {}
    normalized = (scores-min(scores))/(max(scores)-min(scores))
    sortedScored = list(reversed(sorted(normalized)))
    for i in range (len (normalized)):
        sentence_map[i] = [sentences[i], normalized[i], sortedScored.index(normalized[i])]
    return sentence_map

word_embeddings = {}
if not(word_embeddings):
    print("Loading Big File...")
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    print("Finished Loading Big File.")

def textRank(filtered_sentences, sentences):
    sentence_vectors = []
    for i in filtered_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])

    from sklearn.metrics.pairwise import cosine_similarity

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]   
    import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    maxScore = max(scores.values())
    for key in scores:
        scores[key] = scores[key] / maxScore
    sent_map = {}
    sortedScores = list(reversed(sorted(scores.values())))
    for index in range(len(sentences)):
        sent_map[index] = [sentences[index] , scores[index], sortedScores.index(scores[index])]
    return sent_map

### ENSEMBLE WORK ██████████████████████████████████████████████████████████████
def createEnsembledic(tm_dic,luhn_dic,lsa_dic,tr_dic):
    ensemble_dic = {}
    finalScores = []
    for key in tm_dic:
        final_score = tm_dic[key][1] + luhn_dic[key][1]+ lsa_dic[key][1] + tr_dic[key][1]
        finalScores.append(final_score)
        scores_arr = [tm_dic[key][1],luhn_dic[key][1],lsa_dic[key][1],tr_dic[key][1]]
        sent = tm_dic[key][0]
        ensemble_dic[key] = [final_score, scores_arr, sent]
    return ensemble_dic,finalScores

def getTextRanks(filtered_sentences, sentences):
    tm_dic = text_matching(filtered_sentences, sentences)
    luhn_dic = luhn_algorithm(filtered_sentences, sentences)
    lsa_dic = lsa_summarization(filtered_sentences, sentences)
    tr_dic = textRank(filtered_sentences, sentences)
    ensemble_dic, ensemble_scores = createEnsembledic(tm_dic, luhn_dic, lsa_dic, tr_dic)
    return ensemble_dic#, ensemble_scores, tm_dic, luhn_dic, lsa_dic, tr_dic

def allCombs(ensemble_dic):
    lst = []
    for i in ensemble_dic:
        tm = ensemble_dic[i][1][0]
        luhn = ensemble_dic[i][1][1]
        lsa = ensemble_dic[i][1][2]
        tr = ensemble_dic[i][1][3]
        lst.append([ensemble_dic[i][0],tm,luhn,lsa,tr,tm+luhn,tm+lsa,tm+tr,lsa+luhn,tr+luhn,lsa+tr,tm+luhn+lsa,tm+luhn+tr,tm+lsa+tr,luhn+lsa+tr])
    combinationScores = pd.DataFrame(lst, columns =['Ensemble','TM','Luhn','LSA','TR','TM Luhn', 'TM LSA', 'TM TR', 'Luhn LSA', 'Luhn TR', 'LSA TR','TM Luhn LSA','TM Luhn TR','TM LSA TR','Luhn LSA TR']) 
    return combinationScores

def buildDF(filtered_sentences, sentences):
    ensemble_dic = getTextRanks(filtered_sentences, sentences)
    df = allCombs(ensemble_dic)
    return df

def summarizeWith(sentences, df, algorithm, percentage):
    sent = int(percentage * len(df))
    summInds = df.nlargest(n=sent, columns=[algorithm])[algorithm].keys()
    summInds = sorted(summInds)
    summ = ""
    for i in summInds:
        summ += sentences[i]
    return summ