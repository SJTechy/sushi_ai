import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import train_data


model = keras.Sequential([
    tf.keras.layers.Dense(units=6, input_shape=( 6,1 )),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=8,activation="softmax") #Spicy Tuna, California, Shrimp, Salmon, Teriyaki, Tempora, Avocado, Dragon
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(train_data, epochs=50)