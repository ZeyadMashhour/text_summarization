import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords





def preprocessing_text_with_spacy(text, lemmatization = False):
    sentences = text

    # Load the model (English) into spaCy
    nlp = spacy.load('en_core_web_sm')

    # Adding 'sentencizer' component to the pipeline
    nlp.add_pipe('sentencizer')

    # Tokenization & Lemmatization
    lemmatized_sentences = []

    doc = nlp(sentences)

    sentences = []
    if(lemmatization):
        for sentence in doc.sents:
            sentences.append(sentence.text)
            lemmatized_sentences.append([token.lemma_ for token in sentence])
    else:
        for sentence in doc.sents:
            sentences.append(sentence.text)
            lemmatized_sentences.append([token.text for token in sentence])



    # Removing Stop Words & Punctuation 
    filtered_sentences = []

    for sentences_group in lemmatized_sentences:
        filtered = ""

        for sentence in sentences_group:
            sentence_doc = nlp(sentence)
            words_of_sentence = [token.text for token in sentence_doc]

            for token in sentence_doc:
                if token.is_stop == False and token.text.isalpha() == True:
                    filtered += token.text + " "

        filtered_sentences.append(filtered)

    return sentences, filtered_sentences



# def process_df(df):
#     rows,columns = df.shape
#     sentences,filtered_sentences =[],[] 
#     for row in range(rows):
#         articles_sentence , filtered_article = preprocessing_text_with_spacy(df.iloc[row,0])
#         sentences.append(articles_sentence)
#         filtered_sentences.append(filtered_article)
    
#     return sentences, filtered_sentences


from tqdm import tqdm
def process_df(df):
    rows,columns = df.shape
    sentences,filtered_sentences =[],[] 
    for row in tqdm(range(rows)):
        articles_sentence , filtered_article = preprocessing_text_with_spacy(df.iloc[row,0])
        sentences.append(articles_sentence)
        filtered_sentences.append(filtered_article)
    
    return sentences, filtered_sentences



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

def preprocessing_with_nltk(text, lemmatization = False):
    sentences = tokenization(text)  # Tokenize the text into sentences
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence .replace('!','.').replace('?','.').replace('\n',' ')
        sentence = sentence .replace('.',' ')
        filtered_sentences.append(sentence)
    filtered_sentences = remove_stop_words(filtered_sentences)





