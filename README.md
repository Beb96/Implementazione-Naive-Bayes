# Implementazione Naive Bayes

File main.py contiene:
 - Il caricamento del trainingset e del testset per ogni test svolto, l'invocazione del metodo naiveBayes contenuto nel file TextClassification.py per ottenere la precisione e la matrice di confusione di ogni classificatore e la stampa dei risultati ottenuti.
 
File TextClassification.py contiene i metodi:
 - naiveBayes: costruisce la matrice sparsa del trainingset e del testset che ricevo in ingresso, calcola la probabilità delle classi in forma logaritmica e invoca i classificatori di Naive Bayes nella versione di Bernoulli e nella versione Multinomiale
 - bernoulliNB: Definisce i passaggi che il classificatore deve svolgere e calcola la precisione e la matrice di confusione del classificatore.
 - bernoulliModel: metodo che allena il classificatore, ovvero calcola le probabilità in forma logaritmica di ogni parola condizionata dalla classe, applicando Laplace smoothing, utilizzando il metodo bernoulliOccurrencesLikelihoods().
 - bernoulliClassifier: Classifica ogni documento del testset con una classe utilizzando la tecnica di massima verosomiglianza
 - multinomialNB: Definisce i passaggi che il classificatore deve svolgere e calcola la precisione e la matrice di confusione del classificatore.
 - multinomialModel: metodo che allena il classificatore, ovvero calcola le probabilità in forma logaritmica di ogni parola condizionata dalla classe, applicando Laplace smoothing, utilizzando il metodo multinomialOccurrencesLikelihoods().
 - multinomialClassifier: Classifica ogni documento del testset con una classe utilizzando la tecnica di massima verosomiglianza.
 
NB: il dataset 20 newsgroup viene caricato dalla libreria sklearn.dataset 

Fonti consultate:

Per l'apprendimento della teoria:
 - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.9324&rep=rep1&type=pdf
 - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

Per l'estrazione documenti dal dataset 20 newsgroup:
 - https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
 - https://scikit-learn.org/stable/datasets/index.html

Per la parte implementativa:
 - https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.html
 - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
 - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
 - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
 - https://scikit-learn.org/stable/modules/feature_extraction.html
 - https://docs.python.org/3/library/collections.html
