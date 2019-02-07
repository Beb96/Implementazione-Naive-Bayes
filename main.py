import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups

from TextClassification import naiveBayes


# Test no.1
# Utilizzo di solo due categorie: alt.atheism, comp.grapichs
print("Test no.1")
categories = ['alt.atheism', 'comp.graphics']
print("Categories: ", categories)

# Caricamento TrainingSet
twenty_train = fetch_20newsgroups(subset = 'train', categories = categories)

# Caricamento TestSet
twenty_test = fetch_20newsgroups(subset='test', categories = categories)

accuracy_bernoulli, conf_matrix_bernoulli, \
accuracy_multinomial, conf_matrix_multinomial = naiveBayes(twenty_train.data, twenty_train.target, twenty_test.data,
                                                           twenty_test.target)

print("Accuracy Bernoulli Naive Bayes: ", accuracy_bernoulli * 100, " %")
print("Bernoulli's confusion matrix")
print(conf_matrix_bernoulli)

print("Accuracy Multinomial Naive Bayes: ", accuracy_multinomial * 100, " %")
print("Multinomial confusion's matrix")
print(conf_matrix_multinomial)

# Test no.2
# Utilizzo delle categorie rec.sport.baseball, misc.forsale, soc.religion.christian
print("Test no.2")
categories = ['rec.sport.baseball', 'misc.forsale', 'soc.religion.christian']
print("Categories: ", categories)

#Caricamento TrainingSet
twenty_train = fetch_20newsgroups(subset='train', categories = categories)

#Caricamento TestSet
twenty_test = fetch_20newsgroups(subset='test', categories = categories)

accuracy_bernoulli, conf_matrix_bernoulli, \
accuracy_multinomial, conf_matrix_multinomial = naiveBayes(twenty_train.data, twenty_train.target, twenty_test.data,
                                                           twenty_test.target)

print("Accuracy Bernoulli Naive Bayes: ", accuracy_bernoulli * 100, " %")
print("Bernoulli's confusion matrix")
print(conf_matrix_bernoulli)

print("Accuracy Multinomial Naive Bayes: ", accuracy_multinomial * 100, " %")
print("Multinomial confusion's matrix")
print(conf_matrix_multinomial)


# Test no.3
# Prendiamo le categorie tutte insieme : rec.sport.baseball, misc.forsale, soc.religion.christian, alt.atheism, comp.grapichs
print("Test no.3")
categories = ['rec.sport.baseball', 'misc.forsale', 'soc.religion.christian','alt.atheism', 'comp.graphics']
print("Categories: ", categories)

# Caricamento TrainingSet
twenty_train = fetch_20newsgroups(subset='train', categories = categories)

#Caricamento TestSet
twenty_test = fetch_20newsgroups(subset='test', categories = categories)

accuracy_bernoulli, conf_matrix_bernoulli, \
accuracy_multinomial, conf_matrix_multinomial = naiveBayes(twenty_train.data, twenty_train.target, twenty_test.data,
                                                           twenty_test.target)

print("Accuracy Bernoulli Naive Bayes: ", accuracy_bernoulli * 100, " %")
print("Bernoulli's confusion matrix")
print(conf_matrix_bernoulli)

print("Accuracy Multinomial Naive Bayes: ", accuracy_multinomial * 100, " %")
print("Multinomial confusion's matrix")
print(conf_matrix_multinomial)

# Test No.4
# Prendiamo tutte le categorie del dataset 20 newsgroup
print("Test no.4")

# Caricamento TrainingSet
twenty_train = fetch_20newsgroups(subset='train')

#Caricamento TestSet
twenty_test = fetch_20newsgroups(subset='test')

accuracy_bernoulli, conf_matrix_bernoulli, \
accuracy_multinomial, conf_matrix_multinomial = naiveBayes(twenty_train.data, twenty_train.target, twenty_test.data,
                                                           twenty_test.target)

print("Accuracy Bernoulli Naive Bayes: ", accuracy_bernoulli * 100, " %")
print("Bernoulli's confusion matrix")
print(conf_matrix_bernoulli)

print("Accuracy Multinomial Naive Bayes: ", accuracy_multinomial * 100, " %")
print("Multinomial confusion's matrix")
print(conf_matrix_multinomial)
