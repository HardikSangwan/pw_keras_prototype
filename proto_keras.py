import random
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

with open("rockyou.txt", "r") as f:
    raw = f.read()
    data = raw.split('\n')

random.shuffle(data)

train_data = data[:1000000]
test_data = data[1000000:2000000]

# integer encode sequences of characters
chars = sorted(list(set(raw)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in train_data:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)

seq_length = 10
dataX = []
dataY = []
n_chars = len(raw)
for i in range(0, n_chars - seq_length, 1):
	seq_in = sequences[i:i + seq_length]
	seq_out = sequences[i + seq_length]
	dataX.append(seq_in)
	dataY.append(seq_out)
n_patterns = len(dataX)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(vocab_size)
y = np_utils.to_categorical(dataY)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))