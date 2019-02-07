from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from collections import Counter, defaultdict

import numpy as np


# Ritorna un dizionario contenente il logaritmo delle probabilità per ogni classe
def occurrences(list):
    no_of_examples = len(list)
    prob = dict(Counter(list)) # Conta il numero di volte che ciascuna classe compare nel train
    for key in prob:
        prob[key] = np.log(prob[key] / no_of_examples)
    return prob


# Calcola il logaritmo di P(word | class) nella versiona di Bernoulli applicando il Laplace smoothing
def bernoulliOccurrencesLikelihoods(list, no_document_classes):

    count = dict(Counter(list))
    nword = 0 # No. documenti contenenti la parola in esame
    for key in count:
        if(key != 0):
            nword += count[key]

    probability = np.log((nword + 1) / (no_document_classes + 2))
    return probability

# Calcola il logaritmo di P(word | class) nella versione multinomiale applicando il Laplace smoothing
def multinomialOccurrencesLikelihoods(list, no_words_classes, no_words_vacublary):

    count = dict(Counter(list))
    nword = 0
    for key in count:
        if(key != 0):
            nword += (key * count[key])

    probability = np.log((nword + 1) / (no_words_classes + no_words_vacublary))
    return probability


# Metodo che ritorna due array di dimensione (numero_classi) * (no_parole_vocabolario)
# log_prob : contiene il logaritmo di P(word | class) per ogni word nel vocabolario e per ogni classe
# log_prob_neg : contiene il logaritmo di (1 - P(word | class)) per ogni word nel vocabolario e per ogni classe
def bernoulliModel(classes, train_counts, train_target):

    rows, cols = train_counts.get_shape()
    log_prob_neg = np.ndarray(shape = (len(classes), cols), dtype = float)
    log_prob = np.ndarray(shape = (len(classes), cols), dtype = float)

    index = 0
    for cls in classes:
        row_indices = np.where(train_target == cls)[0]
        subset = train_counts[row_indices, :]
        row_subset, cols_subset = subset.get_shape()
        for j in range(cols_subset):
            subset_col = subset.getcol(j)
            log_prob[index][j] = bernoulliOccurrencesLikelihoods(subset_col.data, row_subset)
            log_prob_neg[index][j] = np.log(1 - np.exp(log_prob[index][j]))
        index += 1
    return log_prob, log_prob_neg

# Ritorna un dizionario di dimensione (numero_classi) * (no_parole_vocabolario) contenete il logaritmo di P(word | class)
# per ogni word nel vocabolario e per ogni classe
def multinomialModel(classes, train_counts, train_target, no_words_vocabulary):

    log_likelihoods = {}
    for cls in classes:
        log_likelihoods[cls] = defaultdict(float)

    for cls in classes:
        row_indices = np.where(train_target == cls)[0]
        subset = train_counts[row_indices,:] # Ottiene la sottomatrice formata dalle solo righe della classe in esame
        no_words_classes = subset.sum() # Conta il numero delle parole nella classe
        sub_rows, sub_cols = subset.get_shape()
        for j in range(sub_cols):
            subset_col = subset.getcol(j)
            log_likelihoods[cls][j] = multinomialOccurrencesLikelihoods(subset_col.data, no_words_classes, no_words_vocabulary)

    return log_likelihoods


# Definisce il classificatore dell'algoritmo Naive Bayes nella versione di bernoulli
# Calcola la P(Class | X) = ln(P(class)) + sommatoria per i da 1 a |V| di (ln(P(xi | class))^xi + ln(1 - P(xi | class))^(1 -xi))
# Ritorna un vettore contenente l'etichette ottenute applicando la tecnica di MAP all'array results
def bernoulliClassifier(test_counts, classes, log_class_probabilities, log_prob_likelihoods, log_prob_neg_likelihoods):

    rows, cols = test_counts.get_shape()
    results = np.ndarray(shape = (rows, len(classes)), dtype = float)

    sum_log_neg = np.sum(log_prob_neg_likelihoods, axis = 1)
    index_row_test = 0
    for i in range(rows):
        row = test_counts.getrow(i)
        index_cls = 0
        for cls in classes:
            sum = log_class_probabilities[cls]
            index_row = row.indices
            for j in index_row:
                sum += (log_prob_likelihoods[index_cls][j] - log_prob_neg_likelihoods[index_cls][j])
            sum += sum_log_neg[index_cls]
            results[index_row_test][index_cls] = sum
            index_cls += 1
        index_row_test += 1

    predicted = []
    for i in range(rows):
        index_cls = None
        max = float("-inf")
        for cls in classes:
            if(results[i][cls] > max):
                max = results[i][cls]
                index_cls = cls
        predicted.append(index_cls)

    return predicted

# Definisce il classificatore dell'algoritmo di Naive Bayes nella versione multinomiale
# Calcola la P(Class | X) = ln(P(class)) + sommatoria per i da 1 a |V| di ln(P(xi | class) * xi)
# Ritorna un vettore contenente l'etichette ottenute applicando la tecnica di MAP all'array results
def multinomialClassifier(test_counts, classes, log_likelihoods, log_class_probabilities):

    rows, cols = test_counts.get_shape()
    results = {}
    for i in range(rows):
        results[i] = defaultdict(float)

    index_row_test = 0
    for i in range(rows):
        row = test_counts.getrow(i)
        index_row = row.indices
        for cls in classes:
            sum = log_class_probabilities[cls]
            value_data = 0
            for j in index_row:
                sum += (log_likelihoods[cls][j] * row.data[value_data])
                value_data += 1
            results[index_row_test][cls] = sum
        index_row_test += 1

    predicted = []
    for i in range(rows):
        index_cls = None
        max = float("-inf")
        for cls in classes:
            if(results[i][cls] > max):
                max = results[i][cls]
                index_cls = cls
        predicted.append(index_cls)

    return predicted

# Definisce i passi da eseguire per l'applicazione del classificatore di testo Naive Bayes nella versione di bernoulli
# Ritorna la precisione e la matrice di confusione che il classificatore ha ottenuto
def bernoulliNB(classes, class_probabilities, train_counts, test_counts, train_target, test_target):

    #Calcolo probabilità condizionata
    log_prob_likelihoods, log_prob_neg_likelihoods = bernoulliModel(classes, train_counts, train_target)

    predicted = bernoulliClassifier(test_counts,classes,class_probabilities, log_prob_likelihoods, log_prob_neg_likelihoods)

    accuracy = accuracy_score(test_target, predicted)
    conf_matrix = confusion_matrix(test_target, predicted)

    return accuracy, conf_matrix

# Definisce i passi da eseguire per l'applicazione del classificatore di testo Naive Bayes nella versione multinomiale
# Ritorna la precisione e la matricie di confusione che il classificatore ha ottenuto
def multinomialNB(classes, class_probabilities, train_counts, test_counts, train_target, test_target):

    rows, cols = train_counts.get_shape()
    no_word_vocabulary = cols

    #likelihoods: Calcola il logaritmo delle probabilità condizionate di ogni parola
    log_likelihoods = multinomialModel(classes, train_counts, train_target, no_word_vocabulary)

    predicted = multinomialClassifier(test_counts, classes, log_likelihoods, class_probabilities)

    accuracy = accuracy_score(test_target, predicted)
    conf_matrix = confusion_matrix(test_target, predicted)

    return accuracy, conf_matrix

# Costruisce la matrice dei conteggi del train e del test
# Invoca il metodo per calcolare il logaritmo delle probabilità delle classi
# Ritorna le precisioni e le matrici di confusione ottenute applicando i classificatori di Naive Bayes nella versione
# di bernoulli e nella versione multinomiale
def naiveBayes(train_data, train_target, test_data, test_target):

    # Costruisce la matrice di train dei conteggi dai documenti di 20 newsgroup
    count_vector = CountVectorizer(stop_words= 'english', lowercase = True)
    train_counts = count_vector.fit_transform(train_data)

    # Costruisce la matrice di test dei conteggi dai documenti di 20 newsgroup
    test_counts = count_vector.transform(test_data)

    # Identifica quali sono le classi
    classes = np.unique(train_target)

    # Calcola probabilità di ogni classe, P(prior)
    log_class_probabilities = occurrences(train_target)

    print("I'm applying Bernoulli Naive Bayes...")
    accuracy_bernoulli, conf_matrix_bernoulli = bernoulliNB(classes, log_class_probabilities, train_counts,
                                                            test_counts, train_target, test_target)
    print("I'm applying Multinomial Naive Bayes...")
    accuracy_multinomial, conf_matrix_multinomial = multinomialNB(classes, log_class_probabilities, train_counts,
                                                            test_counts, train_target, test_target)

    return accuracy_bernoulli, conf_matrix_bernoulli, accuracy_multinomial, conf_matrix_multinomial
