import string

import nltk
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
initialize_nltk()

STOP_WORDS = set(stopwords.words("english"))

IMPORTANT_WORDS = {"not", "no", "nor"}

CUSTOM_STOP_WORDS = {
    word for word in STOP_WORDS
    if word not in IMPORTANT_WORDS
}


def clean_text(text):
    try:
        if not isinstance(text, str):
            logger.warning("Received non-string input for preprocessing.")
            return ""

        # existing preprocessing code

        # lowerCase
        text = text.lower()
        # print('Lower Case Text: ')
        # print(text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # print('Punctuation Removed Text: ')
        # print(text)

        # tokenize
        tokens = text.split()
        # print('Tokens: ')
        # print(tokens)

        # remove stopwords
        tokens = [
            token
            for token in tokens
            if token not in CUSTOM_STOP_WORDS
        ]
        # print('Stopwords Removed: ')
        # print(tokens)

        # Lemmatize
        tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
        # print('Lemmatized Tokens: ')
        # print(tokens)

        text = " ".join(tokens)
        return text

    except Exception:
        logger.exception("Text preprocessing failed.")
        raise
