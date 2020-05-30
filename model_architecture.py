import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,CuDNNLSTM,Bidirectional,Flatten
from keras-self-attention import SeqSelfAttention

sys.path.insert(0,os.getcwd())

from data_pipeline import *

model=Sequential()

model.add(Bidirectional(CuDNNLSTM(512,
                                  input_shape=(sequence_length,datasize),
                                  return_sequences=True)))
model.add(SeqSelfAttention(activation='sigmoid'))

model.add(CuDNNLSTM(512,return_sequences=True))

model.add(CuDNNLSTM(512,return_sequences=True))

model.add(Flatten())
model.add(Dense(datasize,activation='softmax'))
