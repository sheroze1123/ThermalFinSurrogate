from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from pandas import DataFrame, read_csv
import pdb
from numpy.random import rand
from forward_solver import ForwardSolver
from time import time


def main(argv):
    #  train_ratio = 0.7
    #  dataset_size = 1000
    # inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
    #  outputs_full = np.loadtxt(open("training_data_uh.csv", "rb"), delimiter=",")
    #  print("Total dataset size of {} with training ratio of {:0.2f}".
    #  format(outputs_full.shape[0], train_ratio))

    #  solve_start = time()

    grid_x = 400
    grid_y = 400
    solver = ForwardSolver(grid_x, grid_y, 10)
    #  uh_s = []
    #  fin_params = []

    #  for i in range(dataset_size):
        #  fin_param = np.array(
            #  [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2])
        #  uh = solver.solve(fin_param)
        #  uh_s.append(uh)
        #  fin_params.append(fin_param)

    #  solve_end = time()
    #  print("\nGenerated dataset of size {} with grid size {} x {} in {} seconds\n".format(
        #  dataset_size, grid_x, grid_y, solve_end - solve_start))

    config = tf.estimator.RunConfig(save_summary_steps=10, model_dir='inverse_output')

    #  solver.plot_solution(uh, 'test.png')
    inverse_regressor = tf.estimator.Estimator(
        config = config,
        model_fn=cnn_model, 
        params={"fin_params": 6, "grid_x": grid_x, "grid_y": grid_y})
    logging_hook = tf.train.LoggingTensorHook(
        tensors={"loss_c": "l2_loss"}, every_n_iter=5)
    #  train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #  x={"x": np.array(uh_s)},
        #  y=np.array(fin_params),
        #  batch_size=10,
        #  num_epochs=None,
        #  shuffle=True)
    #  inverse_regressor.train(input_fn=train_input_fn,
                            #  steps=80, hooks=[logging_hook])
    inverse_regressor.train(input_fn=solver.input_fn,
                            steps=20, hooks=[logging_hook])

    eval_result = inverse_regressor.evaluate(input_fn=solver.input_fn, steps=10)
    print(eval_result)

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
    input_layer = tf.reshape(features["x"], [-1, params["grid_x"], params["grid_y"], 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 8],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[10, 10], strides=10)
    # Output of pool1 is of dim [batch_size, 40, 40, 32]

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)
    # Output of pool2 is of dim [batch_size, 8, 8, 64]

    # Dense Layer with dropout
    dim_x = int(params["grid_x"]/50)
    dim_y = int(params["grid_y"]/50)
    pool2_flat = tf.reshape(pool2, [-1, dim_x * dim_y * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=params["fin_params"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    # TODO: rewrite this with forwad solve loss
    # Calculate Loss (for both TRAIN and EVAL modes)
    #  loss = tf.losses.mean_squared_error(labels, logits, name="l2_loss")
    loss = tf.reduce_mean(tf.squared_difference(
        labels, logits), name='l2_loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #  eval_metric_ops = {
        #  "accuracy": tf.metrics.accuracy(
        #  labels=labels, predictions=logits)}
    eval_metric_ops = {}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
