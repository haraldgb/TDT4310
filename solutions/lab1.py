import nltk
from lab_utils import LabPredictor
from models import BigramModel, TrigramModel
from util import filter_word, preprocess

class Lab1(LabPredictor):
    def __init__(self):
        super().__init__()
        corpora = nltk.corpus.brown.words(categories='news')
        # lowercase all words to have more occurrences of n-grams
        self.corpora = [w.lower() for w in corpora if w.isalnum()]
        
        most_common_words = nltk.FreqDist(
            w.lower() for w in self.corpora if filter_word(w)
        ).most_common(4)
        self.most_common = [word for word, _ in most_common_words]
        
        self.model = None
        self.backoff_model = None

    def predict(self, text):
        print(f"Lab1 receiving: {text}")
        if not bool(text):
            return self.most_common

        tokens = preprocess(text)
        print(f"Lab1 preprocessed: {tokens}")

        min_length = 1 if isinstance(self.model, BigramModel) else 2
        if len(tokens) < min_length:
            print("backoff")
            return self.backoff_model.predict(tokens)

        print("predicting")
        return self.model.predict(tokens)

    def train(self):
        self.model = TrigramModel(self.corpora)
        self.backoff_model = BigramModel(self.corpora)
