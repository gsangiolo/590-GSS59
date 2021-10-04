import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, Sequential, optimizers, losses, metrics, callbacks
from tensorflow import matmul
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns

# Number of classes (set to 3 now -- low, med, high?)
n_bins = 3

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target_raw = raw_df.values[1::2, 2]

label_encoder = LabelEncoder()
target_class = label_encoder.fit_transform(pd.cut(target_raw, n_bins, retbins=True)[0])

scaler = StandardScaler()
data = scaler.fit_transform(data)

kfold = KFold(n_splits=10, shuffle=True)

fold = 1

fold_loss = []
fold_accuracy = []

x_train, x_test, y_train, y_test = train_test_split(data, target_class, test_size=0.2)

for train, test in kfold.split(data, target_class):

    model = Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_bins, activation='softmax')
        ])
    
    model.compile(optimizer='rmsprop',
            loss='mean_squared_error',
            metrics=['accuracy'])
    
    history = model.fit(
            data[train],
            target_class[train],
            epochs=100,
            batch_size=128
            )
    
    results = model.evaluate(data[test], target_class[test], verbose=0)
    print('Accuracy is: ' + str(results[1]*100) + ' for fold #' + str(fold))
    fold_accuracy.append(results[1] * 100)
    fold_loss.append(results[0])
    fold += 1

sns.scatterplot(range(1,11), fold_loss)
sns.scatterplot(range(1,11), fold_accuracy)


# From Tensorflow Docs:
def model_builder(hp):
  model = Sequential()
  model.add(layers.Flatten(input_shape=(1, 13)))

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
