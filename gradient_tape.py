import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
import datetime as dt
import tensorflow as tf

tf.enable_eager_execution()

xtrain = np.array([[1.0,2.0,3.0],[1.0,4.0,0.0]])
ytrain = np.array([3.0,10.0])


model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, activation = 'relu', input_shape=(3,)),  # input shape required
  tf.keras.layers.Dense(1)
])


# model = Sequential()
# model.add(Dense(16, activation='relu', input_dim=3))
# model.add(Dense(1, activation='linear'))

model.compile(loss='mse',
              optimizer=tf.train.AdamOptimizer(),
              metrics=['accuracy'])
model.fit(xtrain, ytrain,
          batch_size=16,
          epochs=30)

print(model.predict(xtrain))

l0 = model.layers[0].get_weights()
l1 = model.layers[1].get_weights()


wi = tf.Variable([[1.0,10.0,4.0]])
y = np.array([[3.0]])


def loss(model, x, y):
    y_= model(x)
    return tf.losses.mean_squared_error(y, y_)

def grad_step(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, inputs)

opt = tf.train.AdamOptimizer()
for i in range(1000):
    loss_value, grads = grad_step(model, wi, y)
    opt.apply_gradients(zip([grads], [wi]))
    print(model.predict(wi))

