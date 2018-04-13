# http://www.nltk.org/book/ch06.html

import random
import nltk
from nltk.classify import apply_features

path = "./data/"

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    features['suffix2'] = name[-2:].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

print (gender_features2('John'))

labeled_names = []
with open(path+'male.txt') as fp:
    for name in fp:
        tup = (name.rstrip("\n\r"), 'male')
        labeled_names.append(tup)
with open(path+'female.txt') as fp:
    for name in fp:
        tup = (name.rstrip("\n\r"), 'female')
        labeled_names.append(tup)
random.seed(448)
random.shuffle(labeled_names)

train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(gender_features2(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features2(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features2(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features2(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

for (tag, guess, name) in sorted(errors):
     print ('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
