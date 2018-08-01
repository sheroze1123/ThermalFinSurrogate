from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pandas import DataFrame, read_csv
import pdb
from numpy.random import rand
from forward_solver import ForwardSolver


def main(argv):
    batch_size  = tf.flags.FLAGS.batch_size
    train_steps = tf.flags.FLAGS.train_steps
    eval_steps  = tf.flags.FLAGS.eval_steps

    solver = ForwardSolver(batch_size)

    config = tf.estimator.RunConfig(save_summary_steps=5, model_dir='out_dnn_inverse')

    logging_hook = tf.train.LoggingTensorHook(
        tensors={"loss_c": "l2_loss"}, every_n_iter=5)

    inverse_regressor = tf.estimator.Estimator(
        config = config,
        model_fn=dnn_model, 
        params={"fin_params": 6, "solver":solver, "nodes":solver.nodes})

    inverse_regressor.train(input_fn=solver.train_input_fn,
                            steps=train_steps, hooks=[logging_hook])

    eval_result = inverse_regressor.evaluate(input_fn=solver.eval_input_fn, steps=eval_steps)
    print(eval_result)

    fin_params = [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2]
    uh = solver.solve_noiterp(fin_params)

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':np.array([uh])}, 
            shuffle=False)

    prediction = list(inverse_regressor.predict(input_fn=pred_input_fn))
    print("TF Prediction: {}".format(prediction[0]))
    print("True params: {}".format(fin_params))
    print("Prediction error: {}".format(
        np.linalg.norm(np.array(prediction[0]) - np.array(fin_params))))

def dnn_model(features, labels, mode, params):
    '''
    Deep Neural Network to map nodal values of temperature to thermal 
    conductivities. 

    Arguments:
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
    dense1 = tf.layers.dense(features["x"], units=params["nodes"], activation=tf.nn.relu)

    dense2 = tf.layers.dense(dense1, units=params["nodes"], activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(dropout2, units=params["nodes"], activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=params["fin_params"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    # TODO: rewrite this with forwad solve loss
    # Calculate Loss (for both TRAIN and EVAL modes)
    #  loss = tf.losses.mean_squared_error(labels, logits, name="l2_loss")
    loss = tf.reduce_mean(tf.squared_difference(
        labels, logits), name='l2_loss')
    #  loss =  solver.tf_solve(fin_param_list[0])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.2)
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
    tf.flags.DEFINE_integer('batch_size', 10, 'Number of images to process in a batch.')
    tf.flags.DEFINE_integer('train_steps', 400, 'Number of training steps to take.')
    tf.flags.DEFINE_integer('eval_steps', 10, 'Number of evaluation steps to take.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
