# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0
import collections
from collections import Counter
import csv
from tfidf import *

def create_vocabl(words, numWords, stopwords):
    vocab = {}
    wordcount = {}
    for word in words:
        #filter
        word = filter(word)
        if len(word) < 1:
            continue
        if word not in stopwords:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(numWords):
        vocab[word] = count

    return vocab

def create_vocab_from_file(inpFile, stopWordsFile, numWords):
    words = []
    stopwords = set(line.strip() for line in open(stopWordsFile))
    with open(inpFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            wordsInRow = row[0].split()
            # print (wordsInRow)
            for w in wordsInRow:
                words.append(w.lower())
    # print (words)
    print(words.__len__())
    vocab = create_vocabl(words, numWords, stopwords)
    vocablSet = set(vocab)
    return vocablSet

def filter(word):
    word = word.lower()
    word = word.replace(".", "")
    word = word.replace(",", "")
    word = word.replace(":", "")
    word = word.replace("\"", "")
    word = word.replace("!", "")
    word = word.replace("â€œ", "")
    word = word.replace("â€˜", "")
    word = word.replace("*", "")
    return word

def document_count_features(document, vocab):
    document_words = set(document)
    features = {}
    counts = Counter(document_words)
    for word in vocab:
        # features['contains({})'.format(word)] = (word in document_words)
        features['count({})'.format(word)] = counts.get(word, 0)
        # if counts.get(word, 0) != 0:
        #     print ('count({})'.format(word), counts.get(word, 0))
    return features

def tranform_documents_to_wo_category(documents):
    documents_wo_cat = []
    for doc in documents:
        documents_wo_cat.append(doc[0])
    return documents_wo_cat

def document_tfidf_features(document, idfs, vocab):
    # features:
    # list of tuples:
    #       (  {feature : value, feature : value, ...},  class)
    #
    features = {}
    for word in vocab:
        # tfidf1 = tfidf(document, documents_wo_cat, word)
        tfidf1 = tfidf_fast(idfs, document, word)
        features['tfidf({})'.format(word)] = tfidf1
    return features