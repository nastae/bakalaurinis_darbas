import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
from sklearn.metrics import confusion_matrix

path = r'hmnist_28_28_RGB_800.csv'
df = pd.read_csv(path)

fractions=np.array([0.8,0.2])
df=df.sample(frac=1)
train_set, test_set = np.array_split(
    df, (fractions[:-1].cumsum() * len(df)).astype(int))

y_train=train_set['label']
x_train=train_set.drop(columns=['label'])
y_test=test_set['label']
x_test=test_set.drop(columns=['label'])

columns=list(x_train)

input_shape = (32, 32, 3)
conv_base = tf.keras.applications.VGG16(include_top=False,
                     weights='cifar10',
                     input_shape=input_shape)

for layer in conv_base.layers:
  layer.trainable = False

top_model = conv_base.output
top_model = layers.Flatten(name="flatten")(top_model)
top_model = layers.Dense(4096, activation='relu')(top_model)
top_model = layers.Dense(1072, activation='relu')(top_model)
top_model = layers.Dropout(0.2)(top_model)
output_layer = layers.Dense(7, activation='softmax')(top_model)

model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

start_time = time.time()

history = model.fit(np.asarray(x_train),
                    np.asarray(y_train),
                    validation_split=0.2,
                    batch_size = 32,
                    epochs = 5000,
                    shuffle=True,
                    callbacks=[callback])
training_time = time.time() -start_time
m, s = divmod(training_time, 60)h, m = divmod(m, 60)
training_time = "%d:%02d:%02d" % (h, m, s)
print(training_time)

x_test=np.array(x_test, dtype=np.uint8).reshape(-1,28,28,3)

test_loss, test_acc = model.evaluate(np.asarray(x_test), np.asarray(y_test), verbose=2)
print(test_acc)

pred = model.predict(x_test)
matrix = confusion_matrix(y_test, pred.argmax(axis=1))
print(matrix)
