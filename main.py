import pandas as pd
import tensorflow as tf
import test as t

max_vocabulary = 10000

vectorize_layer = tf.keras.layers.TextVectorization(
    vocabulary=("res/bad-words.csv"),
    max_tokens=max_vocabulary,
    output_mode='int',
    output_sequence_length=75)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    tf.keras.layers.Embedding(max_vocabulary + 1, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

data_set = pd.read_csv("res/train.csv")

train_data = data_set["comment_text"]
train_target = data_set.iloc[:,2:]
train_target = pd.DataFrame(data=list(map(lambda arr: 1 if 1 in arr else 0, train_target.values)))

model_history = model.fit(train_data, train_target, epochs=5, validation_split=0.2)
print(model_history)
model.save('models/model.h5')

test_data = pd.read_csv("res/test.csv")["comment_text"].to_list()

t.test(model, test_data)