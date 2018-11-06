"""
Classification Task Outlined Here:
https://www.kaggle.com/c/PLAsTiCC-2018
"""

# Import Packages
####################################################################
import numpy as np, pandas as pd, tensorflow as tf, xgboost as xgb
from random import shuffle
from tqdm import tqdm
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
import re, pprint, re, string, sklearn, gc
from sklearn.model_selection import train_test_split
from functools import reduce
from datetime import datetime
import sklearn.preprocessing
from operator import itemgetter
from sklearn.preprocessing import StandardScaler

# Import Data
####################################################################
folder_path = "C:/Users/user/Desktop/plasticc/"
train = pd.read_csv(folder_path + "training_set.csv")
train_meta = pd.read_csv(folder_path + "training_set_metadata.csv")
#test = pd.read_csv(folder_path + "test_set.csv")
#test_meta = pd.read_csv(folder_path + "test_set.csv")

# Define Functions
####################################################################
def slice_by_index(lst, indices):
    """slice a list with a list of indices"""
    slicer = itemgetter(*indices)(lst)
    if len(indices) == 1:
        return [slicer]
    return list(slicer)

def shuffle_batch(y, batch_size):
    rnd_idx = np.random.permutation(len(y))
    n_batches = len(y) // batch_size
    batch_list = []
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_list.append([z for z in batch_idx])
    return batch_list

def seconds_to_time(sec):
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def mjd_to_unix(df, mjd_col):
    temp_list = []
    for mjd in df[mjd_col]:
        temp_list.append((mjd - 40587) * 86400)
    return temp_list

def plastic_ts_agg(dat, metadat):
    df_copy = dat.\
    groupby(['object_id', 'mjd', 'passband'], axis = 0, as_index = False).\
    agg({'flux': [np.min, np.max, np.mean],
         'flux_err': [np.min, np.max, np.mean],
         'detected': [np.max]}).\
    sort_values(['object_id', 'mjd', 'passband'], axis = 0).\
    fillna(0)
    df_copy.columns = ['object_id', 'mjd', 'passband', 'min_flux', 'max_flux', 
                       'mean_flux', 'min_flux_err', 'max_flux_err', 'mean_flux_err', 'max_detected']
    df_copy['passband'] = df_copy['passband'].astype(int)
    df_copy['object_id'] = df_copy['object_id'].astype(int)
    output = df_copy.sort_values(['object_id', 'mjd', 'passband'])
    start_tm = output[['object_id', 'mjd']]
    start_tm.columns = ['object_id', 'mjd_start']
    start_tm = start_tm.\
    groupby(['object_id'], as_index = False).\
    agg({'mjd_start':'min'})
    output = pd.merge(output, start_tm, 'left', 'object_id')
    output['unix_mjd'] = mjd_to_unix(output, 'mjd')
    output['unix_mjd_start'] = mjd_to_unix(output, 'mjd_start')
    output['tm_elapsed'] = [np.float64(i) for i in output['unix_mjd'] - output['unix_mjd_start']]
    output['dt'] = [datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S') for i in output['unix_mjd']]
    output['doy'] = [int(datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime('%j')) for i in output['dt']]
    output.drop(['mjd_start', 'unix_mjd', 'unix_mjd_start', 'dt', 'mjd'], inplace = True, axis = 1)
    output = pd.merge(output, metadat, 'inner', 'object_id').fillna(0)
    return output

def pd_to_array_bycol(df, bycol, ycol):
    x_list = []
    y_list = []
    uniq_colvals = [ucv for ucv in set(df[bycol])]
    for ucv in uniq_colvals:
        x_list.append(df[df[bycol] == ucv].drop([bycol, ycol], axis = 1).values.astype('float32'))
        y_list.append(int(df[df[bycol] == ucv].iloc[0,:][ycol]))
    return np.asarray(y_list), np.array(x_list)

def array_resizing(arrays):
    nrow_list = [i.shape[0] for i in arrays]
    max_row_size = max(nrow_list)
    ncol = arrays[0].shape[1]
    temp_list = []
    for arr in arrays:
        num_add_rows = max_row_size - arr.shape[0]
        add_rows = np.zeros((num_add_rows, ncol))
        concat_rows = np.concatenate([add_rows,arr])
        temp_list.append(concat_rows)
    return np.asarray(temp_list)

def replace_with_dict(ar, dic):
    keys = np.array(list(dic.keys()))
    vals = np.array(list(dic.values()))
    ord_keys = keys.argsort()
    return vals[ord_keys[np.searchsorted(keys, ar, sorter = ord_keys)]]

# Execute Data Prep Functions
####################################################################
# Split Train and Validation
train_xy = plastic_ts_agg(train, train_meta)
#test_xy = plastic_ts_agg(test, test_meta)
train_xy, valid_xy = train_test_split(train_xy, test_size = 0.2, random_state = 11062018)
del train; del train_meta; gc.collect()

# Scale Data
x_cols = [c for c in train_xy.columns if c not in ['target', 'object_id']]
scaler = StandardScaler()
temp_train_x = pd.DataFrame(scaler.fit_transform(train_xy[x_cols]),
                            index = train_xy.index,
                            columns = x_cols)
temp_valid_x = pd.DataFrame(scaler.transform(valid_xy[x_cols]),
                            index = valid_xy.index,
                            columns = x_cols)
temp_train_y = train_xy[['target', 'object_id']]
temp_valid_y = valid_xy[['target', 'object_id']]
train_xy = pd.concat([temp_train_y, temp_train_x], axis = 1)
valid_xy = pd.concat([temp_valid_y, temp_valid_x], axis = 1)
del temp_train_x; del temp_valid_x; del temp_valid_y; del temp_train_y; gc.collect()

# Split X and Y, Resize X, Replace Target Values
train_y, train_x = pd_to_array_bycol(train_xy, 'object_id', 'target')
valid_y, valid_x = pd_to_array_bycol(valid_xy, 'object_id', 'target')
train_x_resized = array_resizing(train_x)
valid_x_resized = array_resizing(valid_x)
class_lookup = {6:0, 15:1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13}
train_y = replace_with_dict(np.array(train_y), class_lookup)
valid_y = replace_with_dict(np.array(valid_y), class_lookup)
del train_xy; del valid_xy; gc.collect()

# Class Weights
####################################################################
def inverse_class_weights(y):
    uniq_y = sorted(set(y))
    temp_list = []
    for uy in uniq_y:
        temp_list.append(1 - (np.sum([1 if i == uy else 0 for i in y]) / len(y)))
    return temp_list

class_reweighting = inverse_class_weights(train_y)

# Tensorflow Single Layer RNN
####################################################################
n_epochs = 25
batch_size = 50
batch_i = shuffle_batch(y = train_y, batch_size = batch_size)
n_batches = train_x_resized.shape[0] // batch_size
tf.reset_default_graph()
n_steps = train_x_resized.shape[1]
n_inputs = train_x_resized.shape[2]
n_neurons = 2000
n_outputs = len(set(train_y))
learning_rate = 0.0025

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)
logits = tf.layers.dense(states, n_outputs, kernel_regularizer = tf.contrib.layers.l1_regularizer(.01))
class_weights = tf.constant(class_reweighting)
weights = tf.gather(class_weights, y)
xentropy = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits, weights = weights)                        
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in range(n_batches):
            y_batch = train_y[batch_i[i]]
            x_batch = train_x_resized[batch_i[i]]
            x_batch = x_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train)