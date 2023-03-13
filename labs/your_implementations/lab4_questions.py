# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer
# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
# We want bigrams as well as the words by themselves
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess(data):
    documents = []
    for line in data.readlines():
        review_body = ' '.join(line.split(',')[5:])
        documents.append(review_body)

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    for idx in range(len(documents)):
        documents[idx] = documents[idx].lower()  # Convert to lowercase.
        documents[idx] = tokenizer.tokenize(documents[idx])  # Split into words.
    # Remove numbers, but not words that contain numbers.
    documents = [[token for token in doc if not token.isnumeric() and token not in stop_words] for doc in documents]
    # Remove words that are one or two characters.
    documents = [[token for token in doc if len(token) > 2] for doc in documents]
    documents = [[lemmatizer.lemmatize(token) for token in doc] for doc in documents]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(documents, min_count=20)
    for idx in range(len(documents)):
        for token in bigram[documents[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                documents[idx].append(token)

    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in documents]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    return corpus, dictionary


def review_lda_topifier(corpus, dictionary) -> int:

    # Set training parameters.
    num_topics = 5
    chunksize = 100000
    passes = 20
    iterations = 25
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    return 0


if __name__ == '__main__':
    with open('../../data/amazon_reviews/amazon_appliances_reviews.csv', 'r') as f:
        corpus, dictionary = preprocess(f)
        review_lda_topifier(corpus, dictionary)
