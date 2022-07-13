# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:17:58 2022

@author: AneeshDixit
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import nltk
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import tensorflow as tf


from StoringRevs import Storing

query = input("Enter the product name (please be precise): ")

rev_obj = Storing(query)

rev_df, rev_list = rev_obj.storingRevs()
rev_df['Class'] = "positive"


rev_df.to_csv("Reviews.csv", sep='\t', index=False)
Reviews = pd.read_csv('Reviews.csv', sep='\t')
data = Reviews.loc[:, 'review_text':'Class']


def remove_tags(string):
    result = str(string)
    result = result.lower()
    return result


rev_df['review_text'] = rev_df['review_text'].apply(lambda cw: remove_tags(cw))


stop_words = set(stopwords.words('english'))

rev_df['review_text'] = rev_df['review_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st


rev_df['review_text'] = rev_df.review_text.apply(lemmatize_text)

reviews = rev_df['review_text'].values
labels = rev_df['Class'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    reviews, encoded_labels, stratify=encoded_labels, test_size=0.95)


# Hyperparameters of the model
vocab_size = 3000  # choose based on statistics
oov_tok = ''
embedding_dim = 100
max_length = 200  # choose based on statistics, for example 150 to 200
padding_type = 'post'
trunc_type = 'post'


# tokenize sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index


# convert Test dataset to sequence and pad sequences
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)


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

prediction = loaded_model.predict(test_padded)

pred_labels = []
for i in prediction:
    if i >= 0.8:
        pred_labels.append(5)
    elif i >= 0.6:
        pred_labels.append(4)
    elif i >= 0.4:
        pred_labels.append(3)
    elif i >= 0.2:
        pred_labels.append(2)
    elif i > 0:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
print("Ratings of each reviews")
print(pred_labels)
s=(sum(pred_labels)/len(pred_labels))
print()
print("Result Rating of the processed model")
print(s)
if s==5:
    print("😍😍😍😍😍")
elif s>=4:
    print("😊😊😊😊")
elif s>=3:
    print("😐😐😐")
elif s>=2:
    print("😒😒")
else:
    print("😢")
if s>=3:
    print("Positive product")
else:
    print("Negative product")
