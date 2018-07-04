from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
import pandas as pd
import pdb

def main(argv):
    train_ratio = 0.7
    # inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
    outputs_full =  np.loadtxt(open("training_data_uh.csv","rb"), delimiter=",")
    print("Total dataset size of {} with training ratio of {:0.2f}".format(outputs_full.shape[0],train_ratio))

    COLUMN_TYPES = collections.OrderedDict([
        ("k1", float),
        ("k2", float),
        ("k3", float),
        ("k4", float),
        ("Biot", float),
        ("k_center", float)
        ])
    # input_df = pd.DataFrame(inputs, columns=input_columns, na_values="?")
    input_df = pd.read_csv("training_data_mul.csv", names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES, na_values="?")
    train_x = input_df.sample(frac=train_ratio)
    test_x = input_df.drop(train_x.index)

    sampling_indices = [1, 2, 3, 15, 16, 660, 750, 1000, 1250]
    # sampling_indices = [750]
    outputs_sampled_cols = list(map(str,sampling_indices))
    outputs_sampled = outputs_full[:,sampling_indices]
    y_df = pd.DataFrame(outputs_sampled, columns=outputs_sampled_cols)
    train_y = y_df.loc[train_x.index,:]
    test_y = y_df.drop(train_x.index)

    def make_dataset(batch_size, x, y):

        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((dict(x),y))
            dataset = dataset.batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        return input_fn


    #  train_input_fn = tf.estimator.inputs.pandas_input_fn(train_x, y=train_y, batch_size=10, num_epochs=4, shuffle=False)
    train_input_fn = make_dataset(10, train_x, train_y)
    test_input_fn = make_dataset(10, test_x, test_y)

    feature_columns = list(map(lambda x:tf.feature_column.numeric_column(key=x),COLUMN_TYPES.keys()))

    model = tf.estimator.DNNRegressor(hidden_units=[20, 20, 20, 20], 
            label_dimension=9, 
            feature_columns=feature_columns,
            model_dir='./output',
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=0.1))

    #############################################################
    # Custom Estimator
    #############################################################

    #  classifier = tf.estimator.Estimator(
        #  model_fn=my_model,
        #  params={
            #  'feature_columns': feature_columns,
            #  # Two hidden layers of 10 nodes each.
            #  'hidden_units': [20, 20, 20, 20],
            #  # The model must choose between 3 classes.
            #  'n_classes': 9,
        #  })

    #  net = tf.feature_column.input_layer(features, params['feature_columns'])

    #  for units in params['hidden_units']:
        #  net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    #  logits = tf.layers.dense(net, params['n_classes'], activation=None)

    #  def my_model_fn(features, labels, mode, params):  
        #  '''
       #  features - This is batch_features from input_fn
       #  labels   - This is batch_labels from input_fn
       #  mode     - An instance of tf.estimator.ModeKeys, see below
       #  params   - Additional configuration
        #  '''
        #  return true

    #############################################################

    model.train(input_fn=train_input_fn)

    eval_result = model.evaluate(input_fn=test_input_fn)
    print (eval_result)

    average_loss = eval_result["average_loss"]

    print("Loss: {:2.3f}".format(average_loss))

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
