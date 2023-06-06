import tensorflow as tf
from tensorflow import keras

input_dim = 7
output_dim = 400
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', input_shape=(input_dim,)))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=output_dim, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%

#%%
