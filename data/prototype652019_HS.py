--#import statements

from __future__ import print_function
import os
import pickle
import random
import sys
import time

from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
import numpy as np

--#neural net parameters

BATCH_SIZE = 32
SEQUENCE_LENGTH = 25
LEARNING_RATE = 0.01
DECAY_RATE = 0.97
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2

TEXT_SAMPLE_LENGTH = 500
SAMPLING_FREQUENCY = 1000
LOGGING_FREQUENCY = 1000

--#reading in the data to be supplied for training
with open("input.txt", "r") as f:
    tokens = f.read()

seed = find_random_seeds(tokens)
token_counts = Counter(tokens)
tokens = [x[0] for x in token_counts.most_common()]
token_indices = {x: i for i, x in enumerate(tokens)}
indices_token = {i: x for i, x in enumerate(tokens)}
vocab_size = len(tokens)

indices = []
for token in tokens:
	if token in token_indices:
		indices.append(token_indices[token])
data = np.array(indices, dtype=np.int32)

batch_size = 32
seq_length = 50
seq_step = 10
x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)

--#Creating the model for training
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

--#Using the model for creating samples at specific iterations
model = RNNModel(data_provider.vocabulary_size, batch_size=1, sequence_length=1, hidden_layer_size=HIDDEN_LAYER_SIZE, cells_size=CELLS_SIZE, training=False)
text = model.sample(sess, data_provider.chars, data_provider.vocabulary, TEXT_SAMPLE_LENGTH).encode("utf-8")
output = open(output_file, "a")
output.write("Iteration: " + str(iteration) + "\n")
output.write(text + "\n")
output.write("\n")
output.close()

--#An individual sample
state = sess.run(self.cell.zero_state(1, tf.float32))
text = ""
char = chars[0]
for _ in range(TEXT_SAMPLE_LENGTH):
    x = np.zeros((1, 1))
    x[0, 0] = vocabulary[char]
    feed = {self.input_data: x, self.initial_state: state}
    [probabilities, state] = sess.run([self.probabilities, self.final_state], feed)
    probability = probabilities[0]
    total_sum = np.cumsum(probability)
    sum = np.sum(probability)
    sample = int(np.searchsorted(total_sum, np.random.rand(1) * sum))
    predicted = chars[sample]
    text += predicted
    char = predicted
return text

--#Showcasing guessing accuracy
pwd_in_file = open('rockyou.txt')
pwd_out_file = open('output.txt')

pwd_in = pwd_in_file.read().split('\n')
pwd_out = pwd_out_file.read().split('\n')
matches = set(pwd_in) & set(pwd_out)

pwd_in_file.close()
pwd_out_file.close()

print('Correctly Guessed Passwords: ' + str(len(matches))
print('Total Guesses: ' + str(len(pwd_out)))
print('Accuracy: ' + str(len(matches)/len(pwd_out)))
print('Guessed against: ' + str(len(pwd_in)))
print('Correctly Gusses Passwords Listed: ')
print(matches)

