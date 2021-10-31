from tensorflow import keras
import numpy as np
import importlib
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.utils import to_categorical
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Evaluator():

    def __init__(self, mode='gutenberg'):
        self.cleaner = importlib.import_module('01-clean').Cleaner(mode=mode)
        self.n_books = 0
        self.n_words = 300
        self.model_path = './models'
        self.mode = mode

    def evaluate_model(self):
        self.model = keras.models.load_model(self.model_path)

        self.texts = cleaner.gather_text()

        books = cleaner.split_chunks(write=False)


        y = []
        X = []

        # Doc2Vec Model
        doc_model = Doc2Vec.load('./models/doc2vec.model')

        for name, chunks in books.items():
            n_books += 1
            for chunk in chunks:
                y.append(name)
                X.append(doc_model.infer_vector(chunk))


        X = np.array([np.array(xi) for xi in X])
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = to_categorical(y)

        loss, acc, mse = model.evaluate(X, y)

        print('Test Accuracy:', acc)

        y_pred = model.predict(X)

        # From SKLearn Website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
        # Get ROC and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_books):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Write metrics to screen
        print('Metrics:\n')
        print('\nAccuracy: ' + str(acc))
        print('\nLoss: ' + str(loss))
        print('\nMSE: ' + str(mse))
        print('\n\nAUC:\n')
        for i in range(self.n_books):
            print('\nClass: ' + str(i) + ' AUC: ' + str(auc(fpr[i], tpr[i])))
        return
