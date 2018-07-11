import collections
import numpy as np
import tensorflow as tf
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt

def main(argv):
    train_ratio = 0.7
    # inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
    outputs_full =  np.loadtxt(open("training_data_uh.csv","rb"), delimiter=",")
    print("Total dataset size of {} with training ratio of {:0.2f}".
            format(outputs_full.shape[0],train_ratio))

    COLUMN_TYPES = collections.OrderedDict([
        ("k1", float),
        ("k2", float),
        ("k3", float),
        ("k4", float),
        ("Biot", float),
        ("k_center", float)
        ])
    # input_df = pd.DataFrame(inputs, columns=input_columns, na_values="?")
    input_df = pd.read_csv("training_data_mul.csv", 
            names=COLUMN_TYPES.keys(), 
            dtype=COLUMN_TYPES, na_values="?")
    train_x = input_df.sample(frac=train_ratio)
    pred_x = input_df.sample(frac=0.0025)
    test_x = input_df.drop(train_x.index)

    pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=pred_x, num_epochs=1, shuffle=False)

    # TODO: More sensible sampling
    sampling_indices = [1, 2, 3, 15, 16, 660, 750, 1000, 1250]
    # sampling_indices = [750]
    outputs_sampled_cols = list(map(str,sampling_indices))
    outputs_sampled = outputs_full[:,sampling_indices]
    y_df = pd.DataFrame(outputs_sampled, columns=outputs_sampled_cols)
    train_y = y_df.loc[train_x.index,:]
    pred_y = y_df.loc[pred_x.index,:]
    test_y = y_df.drop(train_x.index)

    def make_dataset(batch_size, x, y):

        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((dict(x),y))
            dataset = dataset.batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        return input_fn


    #  train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, batch_size=10, num_epochs=4, shuffle=False)
    train_input_fn = make_dataset(10, train_x, train_y)
    test_input_fn = make_dataset(10, test_x, test_y)

    feature_columns = list(map(lambda x:tf.feature_column.numeric_column(key=x),COLUMN_TYPES.keys()))

    #  model = tf.estimator.DNNRegressor(hidden_units=[20, 20, 20, 20], 
            #  label_dimension=9, 
            #  feature_columns=feature_columns,
            #  model_dir='./output',
            #  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=0.1))

    #############################################################
    # Custom Estimator
    #############################################################

    def custom_model(features, labels, mode, params):
        '''
        features - This is batch_features from input_fn
        labels   - This is batch_labels from input_fn
        mode     - An instance of tf.estimator.ModeKeys, see below
        params   - Additional configuration
        '''

        net = tf.feature_column.input_layer(features, params['feature_columns'])

        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        logits = tf.layers.dense(net, params['n_classes'], activation=None)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

        # Compute loss
        # TODO: Smarter loss
        loss = tf.losses.mean_squared_error(labels, logits) 

        # TODO: Other metrics go here
        #  accuracy = tf.metrics.accuracy(labels=labels,
                                   #  predictions=predicted_classes,
                                   #  name='acc_op')
        #  metrics = {'accuracy': accuracy}
        mean_loss = tf.metrics.mean(loss)
        metrics = {'mean_loss':mean_loss}
        tf.summary.scalar('mean_loss',mean_loss[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    model = tf.estimator.Estimator(
            model_fn=custom_model,
            model_dir='./output',
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=0.1),
            params={
                'feature_columns': feature_columns,
                'hidden_units': [20,20,20,20],
                'n_classes':9
            })

    #############################################################

    model.train(input_fn=train_input_fn)

    eval_result = model.evaluate(input_fn=test_input_fn)
    print (eval_result)

    loss = eval_result["loss"]

    print("Loss: {:2.3f}".format(loss))

    preds = list(model.predict(input_fn=pred_input_fn))
    train_indices = pred_x.index.values
    for i in range(len(preds)):
        train_y = pred_y.loc[train_indices[i],:].values
        plt.plot(train_y)
        plt.plot(preds[i])
        plt.legend(["true","NN"])
        plt.savefig("comp_{}.png".format(i))
        plt.cla()
        plt.clf()
        #  print ('pred: {}'.format(preds[i]))
        #  print ('train_y: {}'.format(train_y))

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
