import nltk
import spacy
from spacy.symbols import nsubj, VERB
from spacy import displacy
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer


def q1():
    sentence = 'The quick brown fox jumps over the lazy dog.'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    noun_phrases = [chunk for chunk in doc.noun_chunks]
    verb_phrases = [possible_subject.head for possible_subject in doc
                    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB]
    print(f'Noun phrases in the sentence: {noun_phrases}')
    print(f'Verb phrases in the sentence: {verb_phrases}')
    return 0


def q2():
    sentence = 'The quick brown fox jumps over the lazy dog.'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    # displacy.serve(doc, style='dep')
    subj = None
    obj = None
    for token in doc:
        if 'subj' in token.dep_:
            subtree = list(token.subtree)
            subj = doc[subtree[0].i:subtree[-1].i+1]
        elif 'dobj' in token.dep_:
            subtree = list(token.subtree)
            obj = doc[subtree[0].i:subtree[-1].i+1]
    print(f'subject: {subj}, object: {obj}')
    return 0


def q3():
    def get_synonyms(word):
        synonyms = []
        for synset in wn.synsets(word):
            for word in synset.lemmas():
                synonyms.append(word.name())
        return set(synonyms)
    # print(get_synonyms('crowd'))

    def get_sentiment(sentence):
        def translate_tag(tag):
            if tag[0] == 'J': return wn.ADJ
            elif tag[0] == 'N': return wn.NOUN
            elif tag[0] == 'R': return wn.ADV
            else: return None

        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        score = 0
        lemmatizer = WordNetLemmatizer()
        applied_lemmas = 0
        for word, tag in tagged:
            wn_tag = translate_tag(tag)
            if wn_tag not in [wn.NOUN, wn.ADJ, wn.ADV]:
                continue
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = swn.senti_synsets(lemma, pos=wn_tag)
            try:
                ss = list(synsets)[0]
            except IndexError:
                continue
            score += ss.pos_score() - ss.neg_score()
            applied_lemmas += 1
        return f'positive({score/applied_lemmas})' if score > 0 else f'negative({score/applied_lemmas})'
    text1 = "I liked the movie very much, it was good."
    text2 = "I disliked the movie very much, it was bad."
    text3 = "Well I don't hate it, but it's not the greatest!"
    print(f'The sentence: "{text1}" is perceived to have a: {get_sentiment(text1)} sentiment.')
    print(f'The sentence: "{text2}" is perceived to have a: {get_sentiment(text2)} sentiment.')
    print(f'The sentence: "{text3}" is perceived to have a: {get_sentiment(text3)} sentiment.')


if __name__ == '__main__':
    q1()
    q2()
    q3()
