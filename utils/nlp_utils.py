import logging
import os
import subprocess

import requests
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = 'en_core_web_sm'

nltk.download('vader_lexicon')
SENT = SentimentIntensityAnalyzer()

try:
    logging.info('Initializing SpaCy NLP')
    MODEL = spacy.load(MODEL_NAME)
except OSError as e:
    logger.info('downloading spacy model')
    subprocess.call(('python -m spacy download {}'.format(MODEL_NAME)).split())
    MODEL = spacy.load(MODEL_NAME)
finally:
    logging.info('SpaCy NLP initialized')

PROFANITY_BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'bad-words.txt')

if not os.path.exists(PROFANITY_BLACKLIST_FILE):
    r = requests.get('https://www.cs.cmu.edu/~biglou/resources/bad-words.txt')
    with open(PROFANITY_BLACKLIST_FILE, 'wb') as bad_words_out:
        bad_words_out.write(r.content)

with open(PROFANITY_BLACKLIST_FILE) as profanity_in:
    PROFANITY_BLACKLIST = set([line.strip() for line in profanity_in.readlines() if len(line.strip())])

POS_TAG_BLACKLIST = set(['PROPN', 'PART'])


def get_spacy_nlp():
    return MODEL


def get_sentiment_analyzer():
    return SENT


def is_positive(in_utterance, positive_threshold=0.5, negative_threshold=0.7):
    sentiment_analyzer = get_sentiment_analyzer()
    intent_markers = sentiment_analyzer.polarity_scores(in_utterance)
    return positive_threshold < intent_markers['pos'] and intent_markers['neg'] < negative_threshold


def is_negative(in_utterance, positive_threshold=0.5, negative_threshold=0.7):
    sentiment_analyzer = get_sentiment_analyzer()
    intent_markers = sentiment_analyzer.polarity_scores(in_utterance)
    return negative_threshold < intent_markers['neg'] and intent_markers['pos'] < positive_threshold


def profanity_check(in_utterance, spacy_nlp=get_spacy_nlp()):
    for token in spacy_nlp(in_utterance):
        if token.text in PROFANITY_BLACKLIST:
            return False
    return True


def contains_nes(in_utterance, spacy_nlp=get_spacy_nlp()):
    return len(spacy_nlp(in_utterance).ents) != 0


def contains_blacklisted_pos_tags(in_utterance, spacy_nlp=get_spacy_nlp()):
    for token in spacy_nlp(in_utterance):
        if token.pos_ in POS_TAG_BLACKLIST:
            return True
    return False
