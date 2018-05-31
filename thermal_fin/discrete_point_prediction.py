import numpy as np

inputs = np.loadtxt(open("training_data_mul.csv","rb"), delimiter=",")
outputs_full =  np.loadtxt(open("training_data_uh.csv","rb"), delimiter=",")

sampling_indices = [1, 2, 3, 15, 16, 660, 750, 1000, 1250]
outputs_sampled = outputs_full[:,sampling_indices]

