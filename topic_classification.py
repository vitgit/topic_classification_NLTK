import nltk
from utils import create_vocabl
from utils import filter
from utils import document_count_features
import random
# features:
# list of tuples:
#       (  {feature : value, feature : value, ...},  class)
#
# documents:
# list of tuples with (cat and list of words):
# [([bad, person], neg), ([good, person], pos)]

import csv

# create vocab:
words = []
numWords = 3000
stopwords = set(line.strip() for line in open('./data/stopwords.txt'))
with open('./data/train_r6_test4.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        wordsInRow = row[0].split()
        # print (wordsInRow)
        for w in wordsInRow:
            words.append(w.lower())

# print (words)
print (words.__len__())

vocab = create_vocabl(words, numWords, stopwords)
vocablSet = set(vocab)

# create documents:
# list of tuples with (cat and list of words):
# [([bad, person], neg), ([good, person], pos)]
documents = []
with open('./data/train_r6_test4.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader: # one doc
        content = row[0]
        topicLabel = row[1]
        topicName = row[2]

        words = []
        wordsInRow = content.split()
        for w in wordsInRow:
            w = filter(w)
            if w not in vocablSet:
                continue
            words.append(w)
        if words.__len__() < 1:
            continue
        tup = (words, topicName)
        documents.append(tup)
        random.seed(4)
        random.shuffle(documents)

# print(documents)

featuresets = [(document_count_features(d, vocab), c) for (d,c) in documents]
train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy", nltk.classify.accuracy(classifier, test_set))

