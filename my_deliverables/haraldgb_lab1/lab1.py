import re
from typing import List

import nltk
from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader
from nltk.corpus import brown
from nltk.util import ngrams
from nltk.lm import NgramCounter
from lab_utils import LabPredictor

# pylint: disable=pointless-string-statement
""" Welcome to the first lab!

The comments and TODOs should guide you through the implementation,
but feel free to modify the variables and the overall structure as you see fit.

It is important to ekep the name of the main class: Lab1, as this 
is imported by the `lab_runner.py` file.

You should complete the code for the classes:
- NgramModel (superclass of BigramModel and TrigramModel)
- BigramModel (should be a simple implementation with a few parameters)
- TrigramModel (should be a simple implementation with a few parameters)
- Lab1 (the main logic for parsing input and handling models)
"""


class NgramModel:
    """ The main class for all n-gram models

    Here you will create your model (based on N)
    and complete the predict method to return the most likely words.
    
    """
    def __init__(self, corpus: CategorizedTaggedCorpusReader = brown, n_gram=1, words_to_return=4) -> None:
        """ the init method should load/train your model
        Args:
            n_gram (int, optional): 2=bigram, 3=trigram, ... Defaults to 1.
        """
        print(f"Loading {n_gram}-gram model...")
        self.n_gram = n_gram
        self.words_to_return = words_to_return  # how many words to show in the UI

        # choosing to do this exercise in a different manner, so no model in the nltk sense.
        text_ngrams = list(ngrams(corpus.words(), self.n_gram))
        self.ngram_counts = NgramCounter([text_ngrams])

    def predict(self, tokens: List[str]) -> List[str]:
        """ given a list of tokens, return the most likely next words

        Args:
            tokens (List[str]): preprocessed tokens from the LabPredictor

        Returns:
            List[str]: selected candidates for next-word prediction
        """
        # we're only interested in the last n-1 words.
        # e.g. for a bigram model,
        # we're only interested in the last word to predict the next
        n_tokens = tokens[-(self.n_gram - 1):]

        sorted_distribution = sorted(self.ngram_counts[n_tokens].items(), key=lambda count: count[1])
        return get_most_common_words(sorted_distribution, self.words_to_return)


class BigramModel(NgramModel):
    def __init__(self, corpus=brown) -> None:
        super().__init__(n_gram=2)


class TrigramModel(NgramModel):
    def __init__(self, corpus=brown) -> None:
        super().__init__(n_gram=3)


class Lab1(LabPredictor):
    def __init__(self):
        super().__init__()
        self.corpus = brown
        self.words_to_return = 4

        # Select first words by looking for distribution of words that start sentences in the corpus.
        common_sent_beginnings = nltk.FreqDist([sentence[0].lower() for sentence in brown.sents()])
        self.start_words = get_most_common_words(common_sent_beginnings.most_common(), self.words_to_return)
        self.model = None
        self.backoff_model = None

    @staticmethod
    def preprocess(text: str) -> List[str]:
        """
        Preprocess the input text as you see fit, return a list of tokens.

        - should you consider parentheses, punctuation?
        - lowercase?
        - find inspiration from the course literature :-)
        """
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        return tokens

    def predict(self, input_text):
        if not bool(input_text):  # if there's no input...
            print("No input, using start words")
            return self.start_words
        tokens = self.preprocess(input_text)
        if len(tokens) == 0:
            print("No valid words as input, using start words")
            return self.start_words

        # make use of the backoff model (e.g. bigram)
        too_few = bool(len(tokens) == 1)
        # select the correct model based on the condition
        if not too_few:
            prediction = self.model.predict(tokens)
            if len(prediction) >= 1:
                return prediction
        prediction = self.backoff_model.predict(tokens)
        if len(prediction) >= 1:
            print("Using bigrams to suggest words, either due to lack of data in corpus or too few input words")
            return prediction
        else:
            print("No corresponding data in corpus for neither bigrams nor trigrams, no suggestions")
            return []

    def train(self) -> None:
        """ train or load the models
        add parameters as you like, such as the corpora you selected.
        """
        self.model = TrigramModel(self.corpus)  # TODO: add needed parameters
        self.backoff_model = BigramModel(self.corpus)  # TODO: add needed parameters


def get_most_common_words(freq_dist, word_count: int = 4) -> List[str]:
    words = []
    for freq in freq_dist:
        if len(words) >= word_count:
            break
        match = re.search(r'\w+', freq[0])
        if match:
            words.append(match.string)
        else:
            continue
    return words
