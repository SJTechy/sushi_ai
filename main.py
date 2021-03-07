import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


model = keras.Sequential([
    tf.keras.layers.Dense(units=6, input_shape=( 6,1 )),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=8,activation="softmax") #Spicy Tuna, California, Shrimp, Salmon, Teriyaki, Tempora, Avocado, Dragon
])

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

xs = tf.data.Dataset.from_tensor_slices((
    x,
    y
))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(xs, epochs=500)