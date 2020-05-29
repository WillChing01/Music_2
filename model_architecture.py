import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Flatten

sys.path.insert(0,os.getcwd())

from data_pipeline import *

model=Sequential()

model.add(Bidirectional(LSTM(512,
                             input_shape=(sequence_length,datasize),
                             return_sequences=True)))

model.add(LSTM(512,return_sequences=True))

model.add(LSTM(512,return_sequences=True))

model.add(Flatten())
model.add(Dense(datasize,activation='softmax'))
