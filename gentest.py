import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print (n_chars)
print (n_vocab)

seq_lenght = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_lenght, 1):
    seq_in = raw_text[i:i + seq_lenght]
    seq_out = raw_text[i + seq_lenght]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print (n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_lenght, 1))
X = X / float(n_vocab)

y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weigh"
checkpoint = ModelCheckpoint
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=5)
