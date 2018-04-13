# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0
import collections

def create_vocabl(words, numWords, stopwords):
    vocab = {}
    wordcount = {}
    for word in words:
        #filter
        word = filter(word)
        if word not in stopwords:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(numWords):
        vocab[word] = count

    return vocab

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

def document_features(document, vocab):
    document_words = set(document)
    features = {}
    for word in vocab:
        features['contains({})'.format(word)] = (word in document_words)
    return features