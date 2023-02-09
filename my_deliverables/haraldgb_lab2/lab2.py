from lab_utils import LabPredictor
from solutions.lab1 import TrigramModel

import nltk
from nltk.corpus import brown

import spacy

# pylint: disable=pointless-string-statement
"""
Second Lab! POS tagging with spaCy.

- As for Lab 1, it's up to you to change anything with the structure, 
    as long as you keep the class name (Lab2, inheriting from LabPredictor)
    and the methods (predict, train)


While NLTK has pre-tagged corpora available, I want you to use spaCy to
compute POS tags on just the text (same as Lab 1),
which is more realistic for a real-world application.

An important note:
- Can you compute the POS tag from a single token, or do you need to look at the context (i.e. sentence)?

"""


class Lab2(LabPredictor):
    def __init__(self):
        super().__init__()
        self.words_to_return = 4
        self.corpus = brown.words(categories='news')
        # Reusing strategy for cold start from previous lab.
        common_sent_beginnings = nltk.FreqDist(
            [sentence[0].lower() for sentence in brown.sents() if sentence[0].isalnum()]
        )
        self.start_words = [count[0] for count in common_sent_beginnings.most_common(self.words_to_return)]

        # models are loaded in 'train'
        self.nlp = None
        self.model = None
        self.tag_model = None

    def predict(self, text):
        print(f"Lab2 receiving: {text}")

        input_doc = self.nlp(text)
        input_tokens, input_tags = [token.text for token in input_doc], [token.pos_ for token in input_doc]
        # We don't need the whole ngram, as we already have access to the rest of it
        # We ask for some extra words from the word predictions, so that the ones that are
        word_prediction = self.model.predict(input_tokens, n_words=self.words_to_return * 3, return_ngram=False)
        # Just looking at the tags, what tag is predicted to come next?
        tag_prediction = self.tag_model.predict(input_tags, n_words=self.words_to_return, return_ngram=False)

        # this is the input text with the suggested words from the word model added at the end.
        # We want to analyze the as much of the input as possible to get the best tags
        predicted_docs = [self.nlp(text + word) for word in word_prediction]
        tagged_predicted_docs = [[token.pos_ for token in doc] for doc in predicted_docs]

        good_tag_predictions = []
        bad_tag_predictions = []
        # Words that have a corresponding tag in the tag_prediction is added to the front of the list to be returned.
        # The rest is put at the end.
        for i in range(len(word_prediction)):
            if tagged_predicted_docs[i][-1] in tag_prediction:
                good_tag_predictions.append(word_prediction[i])
                continue
            else:
                bad_tag_predictions.append(word_prediction[i])

        prediction = good_tag_predictions + bad_tag_predictions
        if prediction:
            return prediction[:self.words_to_return]
        else:
            return self.start_words

    def train(self) -> None:
        # creates extra spaces before and/or after all punctuation, but it shouldn't be problematic for our solution
        text = ' '.join(self.corpus)
        self.nlp = spacy.load('en_core_web_sm')
        corp_doc = self.nlp(text)
        corpus_tokens, corpus_tags = [token.text for token in corp_doc], [token.pos_ for token in corp_doc]
        self.model = TrigramModel(corpus_tokens)
        # Making a trigram model for the tags as well, to get a simple read on how common different trigrams of tags are
        self.tag_model = TrigramModel(corpus_tags)
