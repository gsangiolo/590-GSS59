import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, Sequential, optimizers, losses, metrics, datasets, callbacks
from tensorflow import matmul
import keras_tuner as kt

max_words = 10000

train, test = datasets.imdb.load_data(
    path='imdb.npz', num_words=max_words, skip_top=0, maxlen=None, seed=113,
    start_char=1, oov_char=2, index_from=3
)

x_train, y_train = train
x_test, y_test = test

x_total = np.concatenate((x_train, x_test), axis=0)
y_total = np.concatenate((y_train, y_test), axis=0)

data = np.zeros((len(x_total), max_words))
for i, sequence in enumerate(x_total):
    data[i, sequence] = 1
target_class = np.array(y_total).astype('float32')

# Number of classes
n_bins = 2

#data_url = "http://lib.stat.cmu.edu/datasets/boston"
#raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#target_raw = raw_df.values[1::2, 2]

#label_encoder = LabelEncoder()
#target_class = label_encoder.fit_transform(pd.cut(target_raw, n_bins, retbins=True)[0])

#scaler = StandardScaler()
#data = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, target_class, test_size=0.2)

# From the textbook:
class SimpleDense(layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

    def call(self, inputs):
        y = matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

model = Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_bins, activation='softmax')
    ])

model.compile(optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy'])

history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=128
        )

y_pred_raw = model.predict(x_test)
y_pred = list([np.argmax(y_pred_raw[i]) for i in range(len(y_pred_raw))])
a = accuracy_score(y_pred, y_test)
print('Accuracy is: ', a*100)

# From Tensorflow Docs:
def model_builder(hp):
  model = Sequential()
  model.add(layers.Flatten(input_shape=(1, x_train.shape[1])))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(layers.Dense(units=hp_units, activation='relu'))
  model.add(layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

print("Shape: " + str(x_train.shape))

#x_train = x_train.reshape(1, x_train.shape[0] * x_train.shape[1])

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2, batch_size=128, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)
