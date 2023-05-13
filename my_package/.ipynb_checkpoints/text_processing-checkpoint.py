from lexrank.utils.text import tokenize
from lexrank.mappings.stopwords import STOPWORDS
import spacy
from tqdm import tqdm
import pandas as pd
from typing import List
import re
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

import pickle

def clean_text(text):
    """
    Cleans the text by converting to lowercase, removing special characters and numbers, and removing extra whitespace.
    
    Args:
    - text (str): The text to clean
    
    Returns:
    - The cleaned text as a string
    """
    
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z. ]', ' ', text)

    # Remove extra whitespace
    text = re.sub('\s+', ' ', text).strip()

    return text


def process_one_column_df(data, remove_stopwords=True, lemmatize=True, stem=True):
    """
    Processes a single column of a dataframe or a series of text data by cleaning the text and applying spaCy text preprocessing.

    Args:
    - data (pandas.DataFrame or pandas.Series): The data to process. If a dataframe is passed, the first column will be selected.
    - remove_stopwords (bool): Whether or not to remove stopwords. Default is True.
    - lemmatize (bool): Whether or not to lemmatize words. Default is True.
    - stem (bool): Whether or not to stem words. Default is True.

    Returns:
    - A tuple of two lists: a list of sentences and a list of processed sentences.
    """
    
    # Check if input is a dataframe or series
    if isinstance(data, pd.DataFrame):
        # If dataframe, extract the column
        data = data.iloc[:,0]

    rows = len(data)

    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')

    # Adding 'sentencizer' component to the pipeline
    nlp.add_pipe('sentencizer')

    # Initialize empty lists to hold the sentences and processed sentences
    sentences, processed_sentences = [], [] 

    # Loop through each row of the data
    for row in tqdm(range(rows)):
        # Clean the text using the clean_text function
        cleaned_text = clean_text(data.iloc[row])

        # Process the cleaned text using the preprocessing_text_with_spacy function
        articles_sentence, filtered_article = preprocessing_text_with_spacy(cleaned_text, remove_stopwords, lemmatize, stem, nlp)

        # Append the sentences and processed sentences to the appropriate lists
        sentences.append(articles_sentence)
        processed_sentences.append(filtered_article)

    # Return the list of sentences and processed sentences as a tuple
    return sentences, processed_sentences


def preprocessing_text_with_spacy(article, remove_stopwords=True, lemmatize=True, stem=True,nlp = None):
    """
    Preprocesses text using spaCy library
    
    Args:
    article: str, the text to be processed
    remove_stopwords: bool, whether to remove stopwords or not
    lemmatize: bool, whether to lemmatize the text or not
    stem: bool, whether to stem the text or not
    nlp: spaCy model, default is None
    
    Returns:
    article_str_sentences: list, list of string sentences
    processed_sentences: list, list of processed sentences
    """
    
    # Load the model (English) into spaCy if nlp is None
    if nlp == None:
        nlp = spacy.load('en_core_web_sm')
    
    # Create a spaCy Doc object from the text
    doc = nlp(article)

    # Get the sentences from the doc object and convert them to strings
    article_sentences = list(doc.sents)
    article_str_sentences = [sentence.text for sentence in article_sentences]

    # Removing Stop Words & Punctuation 
    processed_sentences = []
    
    # Remove stopwords and lemmatize
    if lemmatize:
        if remove_stopwords:
            lemmatized_sentences = lemmatize_sentence(article_sentences , STOPWORDS['en'])
        else:
            lemmatized_sentences = lemmatize_sentence(article_sentences, [''])
        return article_str_sentences, lemmatized_sentences
    
    # Remove stopwords
    if remove_stopwords:
        processed_sentences = stopwords_removal(article_sentences)
        return article_str_sentences,processed_sentences

    # No lemmatization or stopwords removal
    for sentence in article_sentences:
        # Tokenize the sentence, remove stopwords, and join the tokens back into a string
        tokens = tokenize(sentence.text,stopwords=[''],keep_numbers=True,keep_emails=True,keep_urls=True,)
        cleaned_text = ' '.join(tokens)
        processed_sentences.append(cleaned_text)
    
    return article_str_sentences, processed_sentences

def lemmatize_sentence(article_sentences, stopwords_set: set) -> List[str]:
    """
    Lemmatizes each word in each sentence in the list of article_sentences, removes stopwords, and returns a list
    of the resulting lemmatized sentences.
    
    Parameters:
    article_sentences : List of sentences to be lemmatized.
    stopwords_set (set): Set of stopwords to be removed.
    
    Returns:
    List[str]: List of lemmatized sentences.
    """
    nlp = spacy.load('en_core_web_sm')
    lemmatized_sentences = []
    for sentence in article_sentences:
        # Tokenize the sentence and remove stopwords
        tokens = tokenize(sentence.text,stopwords=stopwords_set,keep_numbers=True,keep_emails=True,keep_urls=True,)
        # Join the remaining tokens into a string
        cleaned_text = ' '.join(tokens)
        # Lemmatize each word in the sentence and join them into a string
        spacy_tokens = nlp(cleaned_text)
        lemm_tokens = [token.lemma_ for token in spacy_tokens]
        lemm_sentence = ' '.join(lemm_tokens)
        lemmatized_sentences.append(lemm_sentence)
    return lemmatized_sentences


def stopwords_removal(article_sentences) -> List[str]:
    """
    Removes stopwords from each sentence in the list of article_sentences and returns a list of the resulting sentences.
    
    Parameters:
    article_sentences : List of sentences to have stopwords removed.
    
    Returns:
    List[str]: List of processed sentences.
    """
    processed_sentences = []
    for sentence in article_sentences:
        # Tokenize the sentence and remove stopwords
        tokens = tokenize(sentence.text,stopwords=STOPWORDS['en'],keep_numbers=True,keep_emails=True,keep_urls=True,)
        # Join the remaining tokens into a string
        filtered_sentences = ' '.join(tokens)
        processed_sentences.append(filtered_sentences)
    return processed_sentences

def combine_lists(lists: List[List[str]]) -> List[str]:
    '''
    Combines elements of a list of lists into a single list of strings.
    Example: [['a','b','c'] , ['1','2','3']] => ['abc' , '123']
    '''
    combined = []
    for lst in lists:
        # Use the join method to join the elements of each list together with no separator
        # and append the resulting string to the combined list
        combined.append(''.join(lst))
    return combined

def save_list_of_lists(data: List[List[str]], file_path: str) -> None:
    '''
    Saves a list of lists to a binary file using the pickle module.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_list_of_lists(file_path: str) -> List[List[str]]:
    '''
    Loads a list of lists from a binary file using the pickle module and returns it.
    '''
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

##################################
#other means to process text
##################################


def filter_sentences(article, remove_stopwords=True, lemmatize=True, stem=True):
    # Clean the article
    article = clean_text(article)

    # Tokenize the article into sentences
    sentences = sent_tokenize(article)

    # Create a list to store the filtered sentences
    filtered_sentences = []

    # Define functions for removing stop words, lemmatizing, and stemming
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        remove_stopwords = lambda x: x.lower() not in stop_words
    if lemmatize:
        lemmatizer = WordNetLemmatizer().lemmatize
    if stem:
        stemmer = PorterStemmer().stem

    # Loop through each sentence and apply the filters
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Remove stopwords if necessary
        if remove_stopwords:
            words = list(filter(remove_stopwords, words))

        # Lemmatize words if necessary
        if lemmatize:
            words = [lemmatizer(word) for word in words]

        # Stem words if necessary
        if stem:
            words = [stemmer(word) for word in words]

        # Join the filtered words back into a sentence
        filtered_sentence = ' '.join(words)

        # Add the filtered sentence to the list
        if filtered_sentence:
            filtered_sentences.append(filtered_sentence)

    return sentences, filtered_sentences


def filter_sentences_spacy(article, remove_stopwords=True, lemmatize=True, stem=True):
    # Clean the article
    article = clean_text(article)

    # Load the English language model
    nlp = spacy.load('en_core_web_sm')

    # Parse the article into sentences using Spacy
    doc = nlp(article)
    sentences = []
    sentences = list(doc.sents)
    sentences = list(map(str,sentences))#change list f spans to list of string

    # Create a list to store the filtered sentences
    filtered_sentences = []

    # Define functions for removing stop words, lemmatizing, and stemming
    if remove_stopwords:
        stop_words = STOP_WORDS
        remove_stopwords = lambda x: x not in stop_words
    if lemmatize:
        lemmatizer = lambda x: x.lemma_
    if stem:
        stemmer = SnowballStemmer('english').stem

    # Loop through each sentence and apply the filters
    for sentence in doc.sents:
        # Tokenize the sentence into words
        words = [token.text for token in sentence]

        # Remove stop words if necessary
        if remove_stopwords:
            words = list(filter(remove_stopwords, words))

        # Lemmatize words if necessary
        if lemmatize:
            words = [lemmatizer(nlp(word)[0]) for word in words]

        # Stem words if necessary
        if stem:
            words = [stemmer(word) for word in words]

        # Join the filtered words back into a sentence
        filtered_sentence = ' '.join(words)

         # Add the filtered sentence to the list
        if filtered_sentence:
            filtered_sentences.append(filtered_sentence)
        else:
            # If the filtered sentence is empty, add an empty sentence
            filtered_sentences.append('')

    return  sentences,filtered_sentences
