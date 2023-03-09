from collections import Counter

import nltk
import spacy
from lab_utils import LabPredictor
from models import TrigramModel
from util import filter_word, preprocess


class Lab2(LabPredictor):
    def __init__(self):
        super().__init__()
        self.corpus = nltk.corpus.brown.words()

        self.start_words = []
        self.model = None
        self.nlp = None
        
    def predict(self, text):
        print(f"Lab2 receiving: {text}")
        if not bool(text):
            return self.start_words

        tokens = preprocess(text)

        predictions = self.model.predict(tokens, n_words=20, return_ngram=True)
        if len(predictions) == 0:
            return []  # just return an empty list if there's no valid preds

        next_ngrams = [" ".join(ngram) for ngram in predictions]
        next_pos = Counter()
        next_words = []
        for ngram_doc in self.nlp.pipe(next_ngrams):
            next_pos[ngram_doc[-1].pos_] += 1
            next_words.append(ngram_doc[-1])

        most_common_pos = next_pos.most_common(1)[0][0]

        filtered_on_pos = [t.text for t in next_words if t.pos_ == most_common_pos]

        print(next_pos)
        print(most_common_pos)
        print(next_words)
        
        print("filtered:", filtered_on_pos)
        # include only unique words:
        filtered_on_pos = list(set(filtered_on_pos))

        if len(filtered_on_pos) < 4:
            # add more of the predicted words
            additional_words = [w.text for w in next_words if w.text not in filtered_on_pos]
            # keep max 4 total:
            additional_words = additional_words[:4 - len(filtered_on_pos)]

            return filtered_on_pos + additional_words

        return filtered_on_pos[:4]

    def train(self):
        spacy_model = "en_core_web_sm"
        print(f"Loading spaCy model {spacy_model}")
        self.nlp = spacy.load(spacy_model, disable=["parser", "ner", "textcat"])

        self.model = TrigramModel(self.corpus)
        N = 4
        words_filtered = [w.lower() for w in self.corpus if filter_word(w)]
        self.start_words = [w[0] for w in nltk.FreqDist(words_filtered).most_common(N)]

