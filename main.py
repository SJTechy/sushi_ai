import numpy as np
import tensorflow as tf
from model import model
from data import train_data

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

model.fit(train_data, epochs=50)