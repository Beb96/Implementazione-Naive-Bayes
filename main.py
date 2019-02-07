import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups

from TextClassification import naiveBayes

def buildDataset(categories, max_rows):

    n_categories = len(categories)
    dim_train = int((max_rows * 0.7) / n_categories)
    dim_test = int((max_rows * 0.3) / n_categories)

    train_data = []
    train_target = []
    test_data = []
    test_target = []

    categorie = []
    for cat in categories:
        categorie.append(cat)
        print(cat)
        twenty = fetch_20newsgroups(subset='all', categories = categorie)
        data = twenty.data
        target = twenty.target
        count = 0
        while(count < dim_train):
            i = random.randint(0,len(data) - 1)
            train_data.append(data[i])
            train_target.append(target[i])
            data = np.delete(data, i)
            target = np.delete(target, i)
            count += 1

        count = 0
        while(count < dim_test):
            i = random.randint(0,len(data) - 1)
            test_data.append(data[i])
            test_target.append(target[i])
            data = np.delete(data, i)
            target = np.delete(target, i)
            count += 1

        del categorie[0]

    return train_data, train_target, test_data, test_target


# Test no.1
# Utilizzo di solo due categorie: alt.atheism, comp.grapichs

categories = ['alt.atheism', 'comp.graphics']
print("Categories: ", categories)
twenty_train = fetch_20newsgroups(subset = 'train', categories = categories)

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

#Caricamento TrainingSet
categories = ['rec.sport.baseball', 'misc.forsale', 'soc.religion.christian']
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

categories = ['rec.sport.baseball', 'misc.forsale', 'soc.religion.christian','alt.atheism', 'comp.graphics']
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
