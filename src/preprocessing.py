import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans a Jira ticket summary for NLP processing.

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    6. Join tokens back into a sentence
    """    
    
   #  print('Original Text: ')
   #  print(text)
    #lowerCase
    text = text.lower()
    # print('Lower Case Text: ')
    # print(text)
    
    #Remove punctuation
    text = text.translate(
        str.maketrans('','',string.punctuation)
    )
    # print('Punctuation Removed Text: ')
    # print(text)

    #tokenize
    tokens = text.split()
    # print('Tokens: ')
    # print(tokens)

    #remove stopwords
    stop_words = set(stopwords.words("english"))
    important_words = {'not','no','nor'}
    custom_stop_words = [word for word in stop_words if word not in important_words]
    tokens = [token for token in tokens if token not in custom_stop_words]
    # print('Stopwords Removed: ')
    # print(tokens)

    #Lemmatize 
    tokens = [lemmatizer.lemmatize(token,pos='v')
             for token in tokens]
    # print('Lemmatized Tokens: ')
    # print(tokens)

    text = " ".join(tokens)
    return text