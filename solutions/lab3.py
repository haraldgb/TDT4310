import nltk
import pandas as pd
import spacy
from lab_utils import LabPredictor
from models import TrigramModel
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from util import preprocess


def expand_and_filter_sentiment(words, sentiment="positive"):
    """
    based on a list of predicted words,
    filter out the words that correspond to the user-defined sentiment
    """
    positive = set()
    negative = set()
    for word in words:
        synsets = wn.synsets(word)
        if len(synsets) == 0:
            continue
        synset = synsets[0]
        synset_word = synset.name().split(".")[0]
        swn_synset = swn.senti_synset(synset.name())

        if swn_synset.pos_score() > 0:
            positive.add(synset_word)
            positive.add(word)
        elif swn_synset.neg_score() > 0:
            negative.add(synset_word)
            negative.add(word)
    return list(positive) if sentiment == "positive" else list(negative)


class Lab3(LabPredictor):
    def __init__(self):
        super().__init__()
        self.start_words = []
        self.model = None
        self.nlp = None
        self.num_words_to_return = 6
        
    def predict(self, text):
        print(f"Lab3 receiving: {text}")
        if not bool(text):
            return self.start_words

        tokens = preprocess(text)
        print("tokens: ", tokens)

        predictions = self.model.predict(tokens, n_words=self.num_words_to_return, return_ngram=False)
        if len(predictions) == 0:
            return []  # just return an empty list if there's no valid preds

        # expand preds with negative/positive words
        # get the POS tag of the last word
        # and alter the sentiment of any verbs
        words_to_expand = []
        print("predictions: ", predictions)
        for pred in predictions:
            doc = self.nlp(" ".join(tokens + [pred]))
            last_pos = doc[-1].pos_
            if last_pos == "VERB":
                words_to_expand.append(pred)
        print("words to expand: ", words_to_expand)
        expanded = expand_and_filter_sentiment(words_to_expand, sentiment="positive")
        words_to_add = self.num_words_to_return - len(expanded)
        print("expanded:", expanded)

        preds = predictions[:words_to_add] + expanded
        return list(set(preds))

    def train(self):
        # read the "lab3_twitter_all.csv" file to train
        twitter_data = "/Users/tollef/Downloads/git/COURSES/TDT4310/labs/solutions/exercise_solutions/data/lab3_twitter_all.csv"
        twitter_df = pd.read_csv(twitter_data)
        tweets = twitter_df["tweet"].tolist()
        tweets = [t for t in tweets if isinstance(t, str) and len(t) > 20]
        tokenized = [nltk.word_tokenize(tweet) for tweet in tweets]
        words = [w.lower() for tweet in tokenized for w in tweet]

        spacy_model = "en_core_web_sm"
        print(f"Loading spaCy model {spacy_model} for POS tagging")
        self.nlp = spacy.load(spacy_model, disable=["parser", "ner", "textcat"])

        self.model = TrigramModel(words)
        N = 4
        self.start_words = [
            w[0] for w in nltk.FreqDist(words)
            .most_common(N)]

