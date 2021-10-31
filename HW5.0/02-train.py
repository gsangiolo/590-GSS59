import os, math
import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.utils import to_categorical

class Trainer():

    def __init__(self, len_vector=300, epochs=100, n_layers=5, activations=None, mode='gutenberg'):
        # Misnomer -- really the number of classes! Initialize to zero to figure out dynamically how many to use
        self.n_books = 0
        # Misnomer -- really the length of the trained vector!
        self.n_words = len_vector
        self.epochs = epochs
        self.n_layers = n_layers
        self.mode = mode
        if activations is None:
            self.activations = ['relu' for i in range(n_layers)]
        else:
            self.activations = activations

    def read_clean_books_from_dir(self, dir_name='./data_clean/'):
        books = {}
        names = []
        texts = []
        dir_name += self.mode + '/'
        for filename in os.listdir(dir_name):
            with open(os.path.join(dir_name, filename)) as file:
                book = json.load(file)
                books[filename] = book

        for name, chunks in books.items():
            self.n_books += 1
            for chunk in chunks:
                names.append(name)
                texts.append(' '.join(chunk))
        return (names, texts)

    def split_train_val(self, names, texts):
        docs = []
        count = 0
        
        for text in texts:
            docs.append(TaggedDocument(words=text.split(), tags=[str(count)]))
            count += 1
        
        # Doc2Vec Model
        doc_model = Doc2Vec(vector_size=self.n_words, window=5, min_count=5, workers=4, epochs=20)
        doc_model.build_vocab(docs)

        print("Training Doc2Vec model...")
        doc_model.train(docs, total_examples=doc_model.corpus_count, epochs=20)

        doc_model.save('./models/doc2vec.model')
        
        print("Doc2Vec model trained!")

        X_train, X_test, y_train, y_test = train_test_split(doc_model.docvecs, names, test_size=0.3)


        X_train = np.array([np.array(xi) for xi in X_train])
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.array([np.array(xi) for xi in X_test])
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_train = to_categorical(y_train)
        y_test = label_encoder.transform(y_test)
        y_test = to_categorical(y_test)

        return X_train, X_test, y_train, y_test

#print(X_train.shape)
#print(y_train)

#import sys
#sys.exit(1)


    def build_model(self, X_train, y_train, X_val, y_val):
        model = Sequential()

        model.add(SimpleRNN(units=32, input_shape=(1, self.n_words), activation='relu'))
        for i in range(math.floor(self.n_layers/2)):
            model.add(Dense(2**(i+5), activation=self.activations[i]))
            model.add(Dropout(0.2))
        for i in range(math.floor(self.n_layers/2)):
            model.add(Dense(2**(self.n_layers - i+5), activation=self.activations[i]))
        model.add(Dense(self.n_books, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
        model.summary()
        
        history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_val, y_val))
        model.save('./models/')
        return model, history

# From the last homework, which is from the Lecture Codes!
#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
    def evaluate_model(self, model, history, X_train, y_train, X_test, y_test):
#        print(model.evaluate(X_train, y_train))
        train_loss, train_acc, train_mse = model.evaluate(X_train, y_train)
        #val_loss, val_acc = model.evaluate(val_images, val_labels, batch_size=floor(validation_percentage*batch_size))
        test_loss, test_acc, test_mse = model.evaluate(X_test, y_test)
        print('train_acc:', train_acc)
        #print('val_acc: ', val_acc)
        print('test_acc:', test_acc)

        y_pred = model.predict(X_test)
        
        # From SKLearn Website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
        # Get ROC and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_books):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Write metrics to log file 
        with open('log.txt', 'w') as logfile:
            logfile.write('Metrics:\n')
            logfile.write('\nTrain Accuracy: ' + str(train_acc))
            logfile.write('\nVal Accuracy:   ' + str(test_acc))
            logfile.write('\nTrain Loss: ' + str(train_loss))
            logfile.write('\nVal Loss:   ' + str(test_loss))
            logfile.write('\nTrain MSE: ' + str(train_mse))
            logfile.write('\nVal MSE:   ' + str(test_mse))
            logfile.write('\n\nAUC:\n')
            for i in range(self.n_books):
                logfile.write('\nClass: ' + str(i) + ' AUC: ' + str(auc(fpr[i], tpr[i])))


        #BASIC PLOTTING
        I_PLOT=True
        #if(k==N_KFOLD-1): I_PLOT=True
        if(I_PLOT):
            #LOSS
            loss_fig = plt.figure()
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            loss_fig.savefig('./img/loss_val_loss.png', dpi=loss_fig.dpi)
        
            #METRICS
            metrics = ['accuracy', 'mse']
            for metric in metrics:
                metrics_fig = plt.figure()
                plt.clf()
                MT = history.history[metric]
                MV = history.history['val_'+metric]
                plt.plot(epochs, MT, 'bo', label='Training '+metric)
                plt.plot(epochs, MV, 'b',  label='Validation '+metric)
                plt.title('Training and validation '+metric)
                plt.xlabel('Epochs')
                plt.ylabel(metric)
                plt.legend()
                plt.show()
                metrics_fig.savefig('./img/metrics' + metric + '.png', dpi=metrics_fig.dpi)

            # ROC
            # From SKLearn website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
            # Plot ROC Curve for all classes on the same plot, in addition to the macro and micro-averages
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_books)]))
            
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_books):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.n_books

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
            # Plot all ROC curves
            roc_fig = plt.figure()
            plt.clf()
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )
            
            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "gold", "azure", "turquoise", 'indigo', 'olive', 'coral', 'peru'])
            for i, color in zip(range(self.n_books), colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                )

            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            plt.show()

            roc_fig.savefig('./img/roc_curve.png', dpi=roc_fig.dpi)
        return
