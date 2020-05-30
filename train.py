import numpy as np
import sys
import os
import random
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras import optimizers

sys.path.insert(0,os.getcwd())

from data_pipeline import *
from model_architecture import *

total_epochs=500
data_batchsize=64
stop_patience=10
lr_patience=1

start_epoch=0

def step_decay(epoch):
    initial_lrate=0.001
    decay=0.9
    return initial_lrate*(decay**epoch)

train_data,validation_data,test_data=get_data()

train_generator=Data_Generator(train_data,data_batchsize)
validation_generator=Data_Generator(validation_data,data_batchsize,training=False)

opt=optimizers.Adam(learning_rate=0,clipnorm=5)

model.compile(loss='categorical_crossentropy',optimizer=opt)

a=model.predict(np.zeros((1,sequence_length,datasize)))

model.fit(np.zeros((1,sequence_length,datasize)),np.zeros((1,datasize)),epochs=1,verbose=0)

try:
    model.load_weights('weights.hdf5')
    print("Successfully loaded previous weights.")
except:
    print("Cannot load weights. Generating new model.")

pathname=os.path.abspath('weights.hdf5')
checkpoint=ModelCheckpoint(pathname,period=1,monitor='val_loss',
                           verbose=1,save_best_only=True,mode='min')

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=stop_patience)

lrate=LearningRateScheduler(step_decay)

callbacks_list=[checkpoint,es,lrate]

model.fit_generator(train_generator,
                    steps_per_epoch=100000,
                    initial_epoch=start_epoch,
                    epochs=total_epochs,
                    callbacks=callbacks_list,
                    validation_data=validation_generator,
                    validation_steps=10000,
                    workers=8,
                    use_multiprocessing=True)
