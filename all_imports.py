#summarization_algorithm

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

#efficiency_scores
from torchmetrics.text.rouge import ROUGEScore
rouge = ROUGEScore()
from pprint import pprint

#preprocessing_algorithms
from lexrank.utils.text import tokenize
from lexrank.mappings.stopwords import STOPWORDS
import spacy


