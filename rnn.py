import numpy as np
import sys
import os
import random
from keras.models import Sequential,save_model,load_model
from keras.layers import Dense,Dropout,CuDNNLSTM,Bidirectional,Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras_self_attention import SeqSelfAttention
from keras import optimizers

sys.path.insert(0,os.getcwd())

from data_pipeline import *

total_epochs=500
data_batchsize=256
stop_patience=10
lr_patience=1

sequence_length=64
datasize=88

start_epoch=0

def step_decay(epoch):
    initial_lrate=0.001
    decay=0.9
    return initial_lrate*(decay**epoch)

train_data,validation_data,test_data=get_data()
random.shuffle(train_data)
random.shuffle(validation_data)
random.shuffle(test_data)

train_generator=Data_Generator(train_data,data_batchsize)
validation_generator=Data_Generator(validation_data,data_batchsize)
test_generator=Data_Generator(test_data,data_batchsize)

model=Sequential()
        
model.add(Bidirectional(CuDNNLSTM(512,
                             input_shape=(sequence_length,datasize),
                             return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Dropout(0.3))

model.add(CuDNNLSTM(512,return_sequences=True))
model.add(Dropout(0.3))

model.add(CuDNNLSTM(512,return_sequences=True))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(datasize,activation='sigmoid'))

opt=optimizers.adam(learning_rate=0.001,clipnorm=5)

model.compile(loss='binary_crossentropy',optimizer=opt)

a=model.predict(np.zeros((1,sequence_length,datasize)))

model.fit(np.zeros((1,sequence_length,datasize)),np.zeros((1,datasize)),epochs=1,verbose=0)

try:
    model.load_weights('weights.hdf5')
except:
    print("Cannot load weights. Generating new model.")

try:
    start_epoch=np.load('start_epoch.npy')
except:
    start_epoch=0

pathname=os.path.abspath('weights.hdf5')
checkpoint=ModelCheckpoint(pathname,period=1,monitor='val_loss',
                           verbose=1,save_best_only=True,mode='min')

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=stop_patience)

lrate=LearningRateScheduler(step_decay)

callbacks_list=[checkpoint,es,lrate]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.steps_per_epoch,
                    epochs=total_epochs,
                    callbacks=callbacks_list,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.steps_per_epoch,
                    workers=8,
                    use_multiprocessing=True,
                    initial_epoch=start_epoch)
