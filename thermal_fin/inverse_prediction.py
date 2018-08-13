from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pandas import DataFrame, read_csv
import pdb
from numpy.random import rand
from forward_solver import ForwardSolver
from time import time



def main(argv):
    grid_x      = tf.flags.FLAGS.grid_x
    grid_y      = tf.flags.FLAGS.grid_y
    batch_size  = tf.flags.FLAGS.batch_size
    train_steps = tf.flags.FLAGS.train_steps
    eval_steps  = tf.flags.FLAGS.eval_steps

    solver = ForwardSolver(batch_size, grid_x, grid_y)

    config = tf.estimator.RunConfig(save_summary_steps=10, model_dir='inverse_output')

    logging_hook = tf.train.LoggingTensorHook(
        tensors={"loss_c": "l2_loss"}, every_n_iter=5)

    inverse_regressor = tf.estimator.Estimator(
        config = config,
        model_fn=cnn_model,
        params={"fin_params": 6, "grid_x": grid_x, "grid_y": grid_y, "solver":solver})

    inverse_regressor.train(input_fn=solver.train_input_fn,
                            steps=train_steps, hooks=[logging_hook])

    eval_result = inverse_regressor.evaluate(input_fn=solver.eval_input_fn, steps=eval_steps)
    print(eval_result)

    fin_params = [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2]
    uh = solver.solve(fin_params)

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':np.array([uh])}, 
            shuffle=False)

    prediction = list(inverse_regressor.predict(input_fn=pred_input_fn))
    print("TF Prediction: {}".format(prediction[0]))
    print("True params: {}".format(fin_params))
    print("Prediction error: {}".format(
        np.linalg.norm(np.array(prediction[0]) - np.array(fin_params))))

def cnn_model(features, labels, mode, params):
    '''
    features - This is batch_features from input_fn
    labels   - This is batch_labels from input_fn
    mode     - An instance of tf.estimator.ModeKeys, see below
    params   - Additional configuration
    '''

    batch_size = tf.flags.FLAGS.batch_size
    solver = params["solver"]

    # input_layer shape = [batch_size, height, width, channels]
    # -1 for batch size, which specifies that this dimension should be dynamically
    # computed based on the number of input values in features["x"]
    input_layer = tf.reshape(features["x"], [-1, params["grid_x"], params["grid_y"], 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[4, 4], strides=2)
    # Output of pool1 is of dim [batch_size, 200, 200, 32]

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[8, 8], strides=4)
    # Output of pool2 is of dim [batch_size, 100, 100, 32]

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[8, 8], strides=4)
    # Output of pool2 is of dim [batch_size, 50, 50, 32]


    # Dense Layer with dropout
    #  dim_x = int(params["grid_x"]/8)
    #  dim_y = int(params["grid_y"]/8)
    dim_x = 11
    dim_y = 11

    pool2_flat = tf.reshape(pool3, [-1, dim_x * dim_y * 16])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1000, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=params["fin_params"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    #  fin_param_list = tf.unstack(logits)

    #  rs = 0
    #  for f in fin_param_list:
        #  rs += solver.tf_solve(f)

    beta = 0.1

    # TODO: rewrite this with forwad solve loss
    # Calculate Loss (for both TRAIN and EVAL modes)
    #  loss = tf.losses.mean_squared_error(labels, logits, name="l2_loss")
    loss = tf.reduce_mean(tf.squared_difference(
        labels, logits), name='l2_loss') / batch_size  + beta * tf.norm(logits)
    #  loss =  solver.tf_solve(fin_param_list[0])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # TODO: Per param error values in the metrics

    # Add evaluation metrics (for EVAL mode)
    #  eval_metric_ops = {
        #  "accuracy": tf.metrics.accuracy(
        #  labels=labels, predictions=logits)}
    eval_metric_ops = {}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    solve_start = time()
    tf.flags.DEFINE_integer('batch_size', 10, 'Number of images to process in a batch.')
    tf.flags.DEFINE_integer('grid_x', 400, 'Number of grid points in the x direction.')
    tf.flags.DEFINE_integer('grid_y', 400, 'Number of grid points in the y direction.')
    tf.flags.DEFINE_integer('train_steps', 400, 'Number of training steps to take.')
    tf.flags.DEFINE_integer('eval_steps', 100, 'Number of evaluation steps to take.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
    solve_end = time()
    print("\nGenerated and trained with dataset of size {} with grid size {} x {} in {} seconds\n".format(
        tf.flags.FLAGS.batch_size * tf.flags.FLAGS.train_steps, 
        tf.flags.FLAGS.grid_x, 
        tf.flags.FLAGS.grid_y, 
        solve_end - solve_start))

