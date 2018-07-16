from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from pandas import DataFrame, read_csv
import pdb
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import tri, cm
from numpy.random import rand
from scipy.interpolate import griddata, LinearNDInterpolator


def main(argv):
    train_ratio = 0.7
    # inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
    #  outputs_full = np.loadtxt(open("training_data_uh.csv", "rb"), delimiter=",")
    #  print("Total dataset size of {} with training ratio of {:0.2f}".
          #  format(outputs_full.shape[0], train_ratio))
    Aq_s, Fh, nodes, coor, theta_tri = load_FEM()
    params = np.array([rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2])

    Ah = coo_matrix((nodes,nodes))
    for param, Aq in zip(params, Aq_s):
        Ah = Ah + param * Aq

    uh = spsolve(Ah, Fh)

    triangulation = tri.Triangulation(coor[:,0], coor[:,1], theta_tri)
    interpolator = tri.CubicTriInterpolator(triangulation, uh)
    plt.tripcolor(triangulation, uh)
    plt.colorbar()
    plt.savefig("plots/uh.png", dpi=400)
    print("Solution written to plots/uh.png")


def load_FEM():
    '''
    Loads the FEM matrices in sparse format for the forward solve
    Only 6 RHS matrices corresponding to the parameters loaded currently.
    Data generated from MATLAB, so the indices need to be subtracted by 1.

    Returns:
        Aq_s : Array of Aq sparse matrices each with dimension (nodes,nodes)
        Fh   : Load vector for FEM with dimension (nodes,1)
        nodes: Number of FEM nodes 
        coor: coordinates for plottting
        theta_tri: coordinate indices for the triangulation with dimension (triangles, 3)
    '''

    Aq_s = []

    data_dir = 'matlab_data/'
    Fh = np.loadtxt(data_dir + 'Fh.csv')
    nodes = Fh.shape[0]
    coor = np.loadtxt(data_dir + 'coarse_coor.csv',delimiter=',')
    theta_tri = np.loadtxt(data_dir + 'theta_tri.csv', delimiter=",", unpack=True)
    for i in range(1,7):
        col, row, value = np.loadtxt(data_dir + 'Aq' + str(i) + '.csv', delimiter="\t", unpack=True)
        Aq = coo_matrix((value, (row-1, col-1)), shape=(nodes, nodes))
        Aq_s.append(Aq)

    return Aq_s, Fh, nodes, coor, (theta_tri-1).T

def cnn_model(features, labels, mode, params):
    '''
    features - This is batch_features from input_fn
    labels   - This is batch_labels from input_fn
    mode     - An instance of tf.estimator.ModeKeys, see below
    params   - Additional configuration
    '''

    # input_layer shape = [batch_size, height, width, channels]
    # -1 for batch size, which specifies that this dimension should be dynamically
    # computed based on the number of input values in features["x"]
    input_layer = tf.reshape(features["x"], [-1, 125, 125, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer with dropout
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
