from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pdb
from numpy.random import rand
from forward_solver import ForwardSolver
from time import time

def main(argv):
    batch_size  = tf.flags.FLAGS.batch_size
    train_steps = tf.flags.FLAGS.train_steps
    eval_steps  = tf.flags.FLAGS.eval_steps

    solver = ForwardSolver(batch_size)

    config = tf.estimator.RunConfig(save_summary_steps=10, model_dir='out_surrogate')

    logging_hook = tf.train.LoggingTensorHook(
        tensors={"loss_c": "l2_loss"}, every_n_iter=20)

    regressor = tf.estimator.Estimator(
        config = config,
        model_fn = forward_model,
        params={"nodes": solver.nodes, "solver":solver, "n_fin_params":6})

    regressor.train(input_fn=solver.train_input_fn_fwd,
                            steps=train_steps, hooks=[logging_hook])

    eval_result = regressor.evaluate(input_fn=solver.eval_input_fn_fwd, steps=eval_steps)
    print(eval_result)

    fin_params = [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2]
    uh = solver.solve_noiterp(fin_params)

    pred_input_fn_fwd = tf.estimator.inputs.numpy_input_fn(
            x={'x':np.array([fin_params])}, 
            shuffle=False)

    pred = np.array(list(regressor.predict(input_fn=pred_input_fn_fwd))[0])
    sol_norm = np.linalg.norm(uh)
    pred_norm = np.linalg.norm(pred)

    solver.plot_solution_nointerp(uh, "True_solution.png")
    solver.plot_solution_nointerp(pred, "Interp_solution.png")
    print("Prediction norm: {}".format(pred_norm))
    print("Solution norm: {}".format(sol_norm))

def forward_model(features, labels, mode, params):
    '''
    features - This is batch_features from input_fn
    labels   - This is batch_labels from input_fn
    mode     - An instance of tf.estimator.ModeKeys, see below
    params   - Additional configuration
    '''

    batch_size = tf.flags.FLAGS.batch_size
    solver = params["solver"]
    n_fin_params = params["n_fin_params"]
    nodes = params["nodes"]

    dense1 = tf.layers.dense(features["x"], units=n_fin_params, activation=tf.nn.relu)

    dense2 = tf.layers.dense(dense1, units=n_fin_params*3, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(dropout2, units=n_fin_params*6, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense4 = tf.layers.dense(dropout3, units=600, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(
        inputs=dense4, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense5 = tf.layers.dense(dropout4, units=nodes, activation=tf.nn.relu)
    dropout5 = tf.layers.dropout(
        inputs=dense5, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout5, units=nodes)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    #  fin_param_list = tf.unstack(logits)

    #  rs = 0
    #  for f in fin_param_list:
        #  rs += solver.tf_solve(f)

    # TODO: rewrite this with forwad solve loss
    # Calculate Loss (for both TRAIN and EVAL modes)
    #  loss = tf.losses.mean_squared_error(labels, logits, name="l2_loss")
    loss = tf.reduce_mean(tf.squared_difference(
        labels, logits), name='l2_loss')
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
    tf.flags.DEFINE_integer('batch_size', 10, 'Number of images to process in a batch.')
    tf.flags.DEFINE_integer('train_steps', 4000, 'Number of training steps to take.')
    tf.flags.DEFINE_integer('eval_steps', 100, 'Number of evaluation steps to take.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
