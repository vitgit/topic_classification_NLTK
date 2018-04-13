# http://www.nltk.org/book/ch06.html

from nltk.corpus import movie_reviews
import nltk
# nltk.download()
import random

# documents: list of tuples with (cat and list of words):
# [([bad, person], neg), ([good, person], pos)]
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
print (all_words)

# list of words
word_features = list(all_words)[:2000]
print (word_features)

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

# list of tuples:
#       (  {feature : value, feature : value, ...},  class)
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)