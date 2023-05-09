from lexrank.utils.text import tokenize
from lexrank.mappings.stopwords import STOPWORDS

import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm




def preprocessing_text_with_spacy(article, remove_stopwords=True, lemmatize=True, stem=True):

    # Load the model (English) into spaCy
    nlp = spacy.load('en_core_web_sm')

    # Adding 'sentencizer' component to the pipeline
    nlp.add_pipe('sentencizer')

    doc = nlp(article)

    article_sentences = []
    article_sentences = list(doc.sents)
    article_str_sentences = list(map(str,article_sentences))#change list f spans to list of string

    # Removing Stop Words & Punctuation 
    processed_sentences = []
    
    #remove stopwords and lemmatize
    if lemmatize:
        if remove_stopwords:
            #print("lemmatize and remove stopwords")
            lemmatized_sentences = lemmatize_sentence(article_sentences , STOPWORDS['en'])
        else:
            #print("lemmatize but doesnt remove stopwords")
            lemmatized_sentences = lemmatize_sentence(article_sentences, [''])
        return article_str_sentences, lemmatized_sentences
    
    if remove_stopwords:
        #print("doesnt lemmatize but remove stopwords")
        processed_sentences = stopwords_removal(article_sentences)
        return article_str_sentences,processed_sentences

    #doesnt remove stopword and doesnt lemmatize
    #print("doesnt lemmatize or remove stopwords")
    for sentence in article_sentences:
        tokens = tokenize(sentence.text,stopwords=[''],keep_numbers=True,keep_emails=True,keep_urls=True,)
        cleaned_text = ' '.join(tokens)
        processed_sentences.append(cleaned_text)
    
    return article_str_sentences, processed_sentences


def lemmatize_sentence(article_sentences, stopwords_set):
    nlp = spacy.load('en_core_web_sm')
    lemmatized_sentences = []
    for sentence in article_sentences:
            tokens = tokenize(sentence.text,stopwords=stopwords_set,keep_numbers=True,keep_emails=True,keep_urls=True,)
            cleaned_text = ' '.join(tokens)
            spacy_tokens = nlp(cleaned_text)
            lemm_tokens = [token.lemma_ for token in spacy_tokens]
            lemm_sentence = ' '.join(lemm_tokens)
            lemmatized_sentences.append(lemm_sentence)
    return lemmatized_sentences


def stopwords_removal(article_sentences):
    processed_sentences = []
    for sentence in article_sentences:
        tokens = tokenize(sentence.text,stopwords=STOPWORDS['en'],keep_numbers=True,keep_emails=True,keep_urls=True,)
        filtered_sentences = ' '.join(tokens)
        processed_sentences.append(filtered_sentences)
    return processed_sentences


def process_one_column_df(df,remove_stopwords=True, lemmatize=True, stem=True):
    rows = len(df)
    sentences,processed_sentences =[],[] 
    for row in tqdm(range(rows)):
        cleaned_text = clean_text(df.iloc[row])
        articles_sentence , filtered_article = preprocessing_text_with_spacy(cleaned_text,remove_stopwords, lemmatize, stem)
        sentences.append(articles_sentence)
        processed_sentences.append(filtered_article)
    
    return sentences, processed_sentences



def tokenization(text):
    sentences = sent_tokenize(text)
    # count the number of sentences in the text 
    total_sentences = len(sentences)
    #print("Total number of sentences:", total_sentences)
    return sentences

def remove_stop_words(sentences):
    # define a set of stop words 
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        # tokenize each sentence into words
        word_tokens = word_tokenize(sentence)
        # remove all stop words from each sentence
        filtered_words = [word for word in word_tokens if not word in stop_words]
        # join all filtered words back into a single sentence
        filtered_sentence = ' '.join(filtered_words)
        processed_sentences.append(filtered_sentence)
    return processed_sentences

def preprocessing_with_nltk(text, lemmatize = False):
    sentences = tokenization(text)  # Tokenize the text into sentences
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence .replace('!','.').replace('?','.').replace('\n',' ')
        sentence = sentence .replace('.',' ')
        processed_sentences.append(sentence)
    processed_sentences = remove_stop_words(processed_sentences)


def combine_lists(lists):
    '''
    combines elemnt of list of lists, ex [['a','b','c'] , ['1','2','3']] = ['abc' , '123']
    '''
    combined = []
    for lst in lists:
        combined.append(' '.join(lst))
    return combined


import pickle

def save_list_of_lists(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_list_of_lists(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z. ]', ' ', text)


    # Remove extra whitespace
    text = re.sub('\s+', ' ', text).strip()

    return text

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