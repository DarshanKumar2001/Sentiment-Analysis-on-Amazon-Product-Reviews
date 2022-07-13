import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])


tokenizer = Tokenizer(num_words=3000, oov_token='')
test_sequences = tokenizer.texts_to_sequences(["This is a bad product"])
print(test_sequences)

test_padded = pad_sequences(test_sequences, padding='post', maxlen=200)

prediction = loaded_model.predict(test_padded)
