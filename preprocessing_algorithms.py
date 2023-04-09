from lexrank.utils.text import tokenize
from lexrank.mappings.stopwords import STOPWORDS

import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm




def preprocessing_text_with_spacy(article, lemmatization = False,remove_stopwords = False):

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
    if lemmatization:
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


def process_df(df,lemmatization = False,remove_stopwords = False):
    rows,columns = df.shape
    sentences,processed_sentences =[],[] 
    for row in tqdm(range(rows)):
        articles_sentence , filtered_article = preprocessing_text_with_spacy(df.iloc[row,0],lemmatization,remove_stopwords)
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

def preprocessing_with_nltk(text, lemmatization = False):
    sentences = tokenization(text)  # Tokenize the text into sentences
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence .replace('!','.').replace('?','.').replace('\n',' ')
        sentence = sentence .replace('.',' ')
        processed_sentences.append(sentence)
    processed_sentences = remove_stop_words(processed_sentences)
