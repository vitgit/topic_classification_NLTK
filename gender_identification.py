# http://www.nltk.org/book/ch06.html
# from nltk.corpus import names
import random
import nltk
from nltk.classify import apply_features

path = "./data/"

# def gender_features(word):
#     features = {}
#     # features['first_letter'] = word[0]
#     features['last_letter'] = word[-1]
#     return features

def gender_features(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

print(gender_features('John'))

percent_of_x = lambda x, percentage: x * percentage // 100.

# labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
#                  [(name, 'female') for name in names.words('female.txt')])

# create labeled_names (list of tuples) from ./data/
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
# print (labeled_names)

# list of tuples of dict and string
featuresets = [(gender_features(ii), gender) for (ii, gender) in labeled_names]
sample_size = featuresets.__len__();
print (sample_size)
train_size = int(percent_of_x(sample_size, 70))

# train_set, test_set = featuresets[:train_size], featuresets[train_size:]
train_set = apply_features(gender_features, labeled_names[:train_size])
test_set = apply_features(gender_features, labeled_names[train_size:])

classifier = nltk.NaiveBayesClassifier.train(train_set)
print (classifier.classify(gender_features('Neo')))
print (classifier.classify(gender_features('Trinity')))
print (classifier.classify(gender_features('Yulia')))
print(nltk.classify.accuracy(classifier, test_set))

print(classifier.show_most_informative_features(5))

