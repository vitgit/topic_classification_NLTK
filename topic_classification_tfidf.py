import nltk
from utils import *
import random
import time
import csv

# inpFile = './data/train_small.csv'
inpFile = './data/train_r6_test4.csv'
stopWordsFile = './data/stopwords.txt'
numWords = 3000

def create_documents(inpFile):
    # list of tuples with (cat and list of words):
    # [([bad, person], neg), ([good, person], pos)]
    documents = []
    with open(inpFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:  # one doc
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
    return documents

vocablSet = create_vocab_from_file(inpFile, stopWordsFile, numWords)
documents = create_documents(inpFile)
documents_wo_cat = tranform_documents_to_wo_category(documents)

#------------------------
start_time = time.time()
print (start_time)
#------------------------
idfs = store_idfs(documents_wo_cat, vocablSet)
# features:
# list of tuples:
#       (  {feature : value, feature : value, ...},  class)
#
featuresets = [(document_tfidf_features(d, idfs, vocablSet), c) for (d,c) in documents]
#------------------------
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#------------------------

train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy", nltk.classify.accuracy(classifier, test_set))

