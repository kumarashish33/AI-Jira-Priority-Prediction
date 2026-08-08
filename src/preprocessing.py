import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logging_config import logger

# Download required NLTK resources only if missing
def initialize_nltk():
    """Download required NLTK resources if missing."""

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.warning(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)

    logger.info("NLTK resources initialized.")


lemmatizer = WordNetLemmatizer()

def clean_text(text):
    try:
        if not isinstance(text, str):
            logger.warning("Received non-string input for preprocessing.")
            return ""

        # existing preprocessing code

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

    except Exception:
        logger.exception("Text preprocessing failed.")
        raise