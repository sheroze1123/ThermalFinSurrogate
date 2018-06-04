import numpy as np
import tensorflow as tf
import pandas as pd
import pdb

def main(argv):
    inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
    outputs_full =  np.loadtxt(open("training_data_uh.csv","rb"), delimiter=",")

    input_columns = ['k1','k2','k3','k4','Biot','k_center']
    input_df = pd.DataFrame(inputs, columns=input_columns)
    train_x = input_df.sample(frac=0.7)
    test_x = input_df.drop(train_x.index)

    sampling_indices = [1, 2, 3, 15, 16, 660, 750, 1000, 1250]
    outputs_sampled_cols = list(map(str,sampling_indices))
    outputs_sampled = outputs_full[:,sampling_indices]
    y_df = pd.DataFrame(outputs_sampled, columns=outputs_sampled_cols)
    train_y = y_df.loc[train_x.index,:]
    test_y = y_df.drop(train_x.index)

    def make_dataset(batch_size, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((dict(x),y))

        def input_fn():
            dataset.batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        return input_fn

    train_input_fn = make_dataset(10, train_x, train_y)
    test_input_fn = make_dataset(10, test_x, test_y)

    feature_columns = list(map(lambda x:tf.feature_column.numeric_column(key=x),input_columns))

    # pdb.set_trace()

    model = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)


    model.train(input_fn=train_input_fn)

    eval_result = model.evaluate(input_fn=test_input_fn)

    average_loss = eval_result["average_loss"]

    print("Loss: ${:2.3f}".format(average_loss))

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
