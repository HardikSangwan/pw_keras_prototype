import codecs
import os
import collections
import sys
import numpy as np
import time
import random
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential

output_file = "./data/output.txt"
output = open(output_file, "w")
output.close()

# Neural Net Parameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 25
LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2
RNN_SIZE = 128
NUM_LAYERS = 2
TEXT_SAMPLE_LENGTH = 500

def rnn():    
    #Reading in Data to be used in batches. Also defining other common characteristics of character based models
    with codecs.open("./data/input.txt", "r", encoding="utf-8") as file:
        data = file.read()

    lines = data.split('\n')
    # Take a random sampling of lines
    if len(lines) > 50 * 4:
        lines = random.sample(lines, 50 * 4)
    # Take the top quartile based on length so we get decent seed strings
    lines = sorted(lines, key=len, reverse=True)
    lines = lines[:50]
    # Split on the first whitespace before max_seed_length
    seeds = [line[:50].rsplit(None, 1)[0] for line in lines]

    count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocabulary_size = len(chars)
    vocabulary = dict(zip(chars, range(len(chars))))
    tensor = np.array(list(map(vocabulary.get, data)))
    batches_size = int(tensor.size / (BATCH_SIZE * SEQUENCE_LENGTH))
    tensor = tensor[:batches_size * BATCH_SIZE * SEQUENCE_LENGTH]
    inputs = tensor
    targets = np.copy(tensor)
    targets[:-1] = inputs[1:]
    targets[-1] = inputs[0]
    input_batches = np.split(inputs.reshape(BATCH_SIZE, -1), batches_size, 1)
    target_batches = np.split(targets.reshape(BATCH_SIZE, -1), batches_size, 1)
    
    #Defining the model
    model = Sequential()
    model.add(Embedding(vocabulary_size, HIDDEN_LAYER_SIZE, batch_input_shape=(BATCH_SIZE, None)))
    for layer in range(NUM_LAYERS):
        model.add(LSTM(RNN_SIZE, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocabulary_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    train_model = model

    sample_model = Sequential()
    sample_model.add(Embedding(vocabulary_size, HIDDEN_LAYER_SIZE, batch_input_shape=(1, None)))
    for layer in range(NUM_LAYERS):
        sample_model.add(LSTM(RNN_SIZE, stateful=True, return_sequences=True))
        sample_model.add(Dropout(0.2))
    sample_model.add(TimeDistributed(Dense(vocabulary_size, activation='softmax')))
    sample_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    iteration = 0

    while iteration<=100000:
        pointer = 0
        for batch in range(batches_size):
            inputs = input_batches[pointer]
            targets = target_batches[pointer]
            targets = targets[:, :, np.newaxis]
            pointer+=1
            train_model.fit(inputs, targets, batch_size=BATCH_SIZE, shuffle=False, epochs= 1, verbose=0)
            iteration += 1

            #Saving outputs and showing updates at every 100 iterations
            if iteration % 100 == 0:
                sample_model.set_weights(train_model.get_weights())
                sample_model.reset_states()

                indices_token = {i: x for i, x in enumerate(data)}
                token_indices = {x: i for i, x in enumerate(data)}
                seed = random.choice(seeds)
                indices = []
                tokens = seed
                for token in tokens:
                    if token in token_indices:
                        indices.append(token_indices[token])
                seed_vector = [194, 19, 1, 299, 45, 58, 178, 259, 301, 50, 81, 9, 109, 236, 31, 86, 49, 43, 54, 99, 94]

                for char_index in np.nditer(seed_vector):
                    preds = sample_model.predict(np.array([char_index]), verbose=0)

                sampled_indices = np.array([], dtype=np.int32)

                for i in range(TEXT_SAMPLE_LENGTH):
                    char_index = 0
                    if preds is not None:
                        pred = np.asarray(preds[0][0]).astype(np.float64)
                        # Add a tiny positive number to avoid invalid log(0)
                        pred += np.finfo(np.float64).tiny
                        pred = np.log(pred) / 1.0
                        exp_preds = np.exp(pred)
                        pred = exp_preds / np.sum(exp_preds)
                        probas = np.random.multinomial(1, pred, 1)
                        char_index = np.argmax(probas)
                    sampled_indices = np.append(sampled_indices, char_index)
                    preds = sample_model.predict(np.array([char_index]), verbose=0)

                tokens = [indices_token[index] for index in sampled_indices.tolist()]
                sample = ''.join(tokens)

                text = sample.encode("utf-8")
                output = open(output_file, "a")
                output.write(text + "\n")
                output.close()

                print('{{"metric": "iteration", "value": {}}}'.format(iteration))

if __name__ == '__main__':    
    #Training. Sampling.
    rnn()

    #Print results after Password Guessing
    pwd_in_file = open('data/rockyou.txt')
    pwd_out_file = open('data/output.txt')

    pwd_in = pwd_in_file.read().split('\n')
    pwd_out = pwd_out_file.read().split('\n')
    matches = set(pwd_in) & set(pwd_out)

    pwd_in_file.close()
    pwd_out_file.close()

    print('End Result: ')
    print('Correctly Guessed Passwords: ' + str(len(matches)))
    print('Total Guesses: ' + str(len(pwd_out)))
    print('Accuracy: ' + str(len(matches)/len(pwd_out)))
    print('Guessed against: ' + str(len(pwd_in)))
    print('Correctly Guessed Passwords Listed: ')
    print(matches)
