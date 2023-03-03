from typing import List, Optional, Tuple
import nltk

class NgramModel:
    def __init__(self, measures, finder, corpus, n_gram) -> None:
        self.model = finder.from_words(corpus).score_ngrams(measures.raw_freq)
        self.n_gram = n_gram
        
    def compare_ngram(self, this: Optional[Tuple[str]], that: Tuple[str]) -> bool:
        """ a function to compare the n-1 words in the nltk collocation model
        Args:
            this (str or list of strings): the user input
            that (list): the existing tuples up to range N-1
        """
        if not this:
            return False

        return list(that[0][:len(this)]) == this
    
    def predict(self, tokens: List[str], n_words: int=4, return_ngram: bool = False) -> List[str]:
        n_tokens = tokens[-(self.n_gram - 1):]
        model_name = "bigram" if self.n_gram == 2 else "trigram"
        print(f"{model_name} tokens: {n_tokens}")
        probas = [w for w in self.model if self.compare_ngram(n_tokens, w)]
        best_probas = sorted(probas, key=lambda x: x[1], reverse=True)
        # best_words = [w[0][-1] for w in best_probas]
        if return_ngram:
            return [w[0] for w in best_probas][:n_words]
        return [w[0][-1] for w in best_probas][:n_words]

class BigramModel(NgramModel):
    def __init__(self, corpus) -> None:
        super().__init__(
            measures=nltk.collocations.BigramAssocMeasures(),
            finder=nltk.collocations.BigramCollocationFinder,
            corpus=corpus,
            n_gram=2
        )

class TrigramModel(NgramModel):
    def __init__(self, corpus) -> None:
        super().__init__(
            measures=nltk.collocations.TrigramAssocMeasures(),
            finder=nltk.collocations.TrigramCollocationFinder,
            corpus=corpus,
            n_gram=3,
        )