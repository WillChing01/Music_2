import sys
import os
from keras.models import Sequential
from keras.layers import Dense,CuDNNLSTM,Bidirectional,Flatten
from keras_self_attention import SeqSelfAttention

sys.path.insert(0,os.getcwd())

from data_pipeline import *

model=Sequential()

model.add(Bidirectional(CuDNNLSTM(512,
                                  input_shape=(sequence_length,datasize),
                                  return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))

model.add(CuDNNLSTM(512,return_sequences=True))

model.add(CuDNNLSTM(512,return_sequences=True))

model.add(Flatten())
model.add(Dense(datasize,activation='softmax'))
