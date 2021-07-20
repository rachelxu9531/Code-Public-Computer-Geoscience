
# Startup Google CoLab
try: %tensorflow_version 2.x
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False

#
cely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# Read the data set
df = pd.read_csv('Field_input.csv', encoding="cp1252", na_values=['NA', '?'])

# df['Cut Depth_mm'] = zscore(df['Cut Depth_mm'])
df['ICR_m3/hr'] = zscore(df['ICR_m3/hr'])
df['SE_Mj/m3'] = zscore(df['SE_Mj/m3'])
df['Cutter radius_mm'] = zscore(df['Cutter radius_mm'])
df['Cut Depth_mm'] = zscore(df['Cut Depth_mm'])

print(df)


# Convert to numpy - Classification
x_columns = df.columns.drop('Rock Type')
x = df[x_columns].values
dummies = pd.get_dummies(df['Rock Type']) # Classification
products = dummies.columns
y = dummies.values

print(x)
print(y)

import pandas as pd
import os
import numpy as np
import time
import tensorflow.keras.initializers
import statistics
import tensorflow.keras
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization


def generate_model(dropout, neuronPct, neuronShrink):
    # We start with some percent of 1000 starting neurons on the first hidden layer.
    neuronCount = int(neuronPct * 1000)

    # Construct neural network
    # kernel_initializer = tensorflow.keras.initializers.he_uniform(seed=None)
    model = Sequential()

    # So long as there would have been at least 25 neurons and fewer than 10
    # layers, create a new layer.
    layer = 0
    while neuronCount > 5 and layer < 10:
        # The first (0th) layer needs an input input_dim(neuronCount)
        if layer == 0:
            model.add(Dense(neuronCount,
                            input_dim=x.shape[1],
                            activation=PReLU()))
        else:
            model.add(Dense(neuronCount, activation=PReLU()))
        layer += 1

        # Add dropout after each hidden layer
        model.add(Dropout(dropout))

        # Shrink neuron count for each layer
        neuronCount = neuronCount * neuronShrink

    model.add(Dense(y.shape[1], activation='softmax'))  # Output
    return model


def evaluate_network(dropout, lr, neuronPct, neuronShrink):
    SPLITS = 5

    # Bootstrap
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.5)

    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0

    # Loop through samples
    for train, test in boot.split(x, df['Rock Type']):
        start_time = time.time()
        num += 1

        # Split train and test
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        model = generate_model(dropout, neuronPct, neuronShrink)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                patience=100, verbose=0, mode='auto', restore_best_weights=True)

        # Train on the bootstrap sample
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  callbacks=[monitor], verbose=0, epochs=2000)
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)

        # Predict on the out of boot (validation)
        y_pred = model.predict(x_test)

        # Measure this bootstrap's log loss
        y_compare = np.argmax(y_test, axis=1)  # For log loss calculation
        score = metrics.log_loss(y_compare, y_pred)
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        m2 = statistics.mean(epochs_needed)
        mdev = statistics.pstdev(mean_benchmark)

        # Record this iteration
        time_took = time.time() - start_time
        print(
            f"#{num}: score={score:.7f}, mean score={m1:.7f}, stdev={mdev:.7f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")
    tensorflow.keras.backend.clear_session()
    return (-m1)


from bayes_opt import BayesianOptimization
import time

# Supress NaN warnings
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Bounded region of parameter space
pbounds = {'dropout': (0.0, 0.499), 'lr': (0.0, 1), 'neuronPct': (0.01, 1), 'neuronShrink': (0.01, 1)}

optimizer = BayesianOptimization(f=evaluate_network, pbounds=pbounds, verbose=2, random_state=1)
# verbose = 1 prints only when a maximum
# is observed, verbose = 0 is silent

# start_time = time.time()
optimizer.maximize(init_points=10, n_iter=20)
# time_took = time.time() - start_time

# print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)


def check_network(dropout, lr, neuronPct, neuronShrink):
    SPLITS = 5

    # Bootstrap
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.5)

    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0

    # Loop through samples
    for train, test in boot.split(x, df['Rock Type']):
        start_time = time.time()
        num += 1

        # Split train and test
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        model = generate_model(dropout, neuronPct, neuronShrink)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                patience=100, verbose=0, mode='auto', restore_best_weights=True)

        # Train on the bootstrap sample
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  callbacks=[monitor], verbose=0, epochs=2000)
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)

        # Predict on the out of boot (validation)
        y_pred = model.predict(x_test)

        # Measure this bootstrap's log loss
        y_compare = np.argmax(y_test, axis=1)  # For log loss calculation
        score = metrics.log_loss(y_compare, y_pred)
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        m2 = statistics.mean(epochs_needed)
        mdev = statistics.pstdev(mean_benchmark)

        # Record this iteration
        time_took = time.time() - start_time
        print(
            f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")

    tensorflow.keras.backend.clear_session()
    return (-m1, y_pred, y_test, score, x_test)


out=check_network(dropout= 0.05139030233799062, lr=0.08021990125087118, neuronPct=0.2113077788841243,neuronShrink=0.35100988615110373)
print(out)

def convert(sequence):
    result_array = np.empty((0, len(sequence)))
    for i in range(len(sequence)):
        index= np.argmax(sequence[i])
        if index==0:
            result='HSS'
        elif index ==1:
            result='LSS'
        else:
            result='MSS'
        result_array = np.append(result_array, [result])
    return result_array

df_test = pd.DataFrame({'y_test': convert(out[2]),'norm Cut Depth_mm': out[4][:,0], 'norm ICR_m3/hr': out[4][:,1], 'norm SE_Mj/m3': out[4][:,2], 'norm Cutter radius_mm': out[4][:,3]},
                         columns = ['y_test','norm Cut Depth_mm' , 'norm ICR_m3/hr','norm SE_Mj/m3','norm Cutter radius_mm'])
df_pred=pd.DataFrame({'y_pred': convert(out[1]),'norm Cut Depth_mm': out[4][:,0], 'norm ICR_m3/hr': out[4][:,1], 'norm SE_Mj/m3': out[4][:,2], 'norm Cutter radius_mm': out[4][:,3]},
                         columns = ['y_pred','norm Cut Depth_mm' , 'norm ICR_m3/hr','norm SE_Mj/m3','norm Cutter radius_mm'])
print (df_test)
print(df_pred)

df_test.y_test = pd.Categorical(df_test.y_test)
df_test['y_test'] = df_test.y_test.cat.codes
df_test['y_test']=df_test['y_test'].replace([0,1,2],[2,0,1])
print(df_test)

df_pred.y_pred = pd.Categorical(df_pred.y_pred)
df_pred['y_pred'] = df_pred.y_pred.cat.codes
df_pred['y_pred']=df_pred['y_pred'].replace([0,1,2],[2,0,1])
print(df_pred)




