import tensorflow as tf
import numpy as np

x = np.array([
    [1,0,1,0,1,0],
    [1,1,0,0,1,1],
    [0,0,1,0,1,1],
    [1,1,0,1,0,1],
    [0,1,1,0,1,1],
    [1,1,0,1,0,0]
])

y = np.array([
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1]
])

train_data = tf.data.Dataset.from_tensor_slices((
    x,
    y
))