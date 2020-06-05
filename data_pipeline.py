from mido import MidiFile
import numpy as np
import sys
import os
import glob
import json
import math
from tensorflow.keras.utils import Sequence
import time

sys.path.insert(0,os.getcwd())

"""
Data representation (1-hot vector):

312-size 1-hot vector.
    -88 note on
    -88 note off
    -32 velocity
    -4 pedal (sustain on/off, soft on/off)
    -100 time shift (0.01-1.00 seconds)
    
"""

datasize=312
sequence_length=256

data_trans=[-3,3]
time_trans=[-0.1,-0.075,-0.05,-0.025,0,0.025,0.05,0.075,0.1]

class Data_Generator(Sequence):

    def __init__(self,filenames,batch_size,training=True):
        #filename - [[filename[0],size[0]],[filename[1],size[1]],...]
        self.filenames=sorted(filenames,key=lambda x:x[1],reverse=True)
        self.batchsize=batch_size
        self.training=training

        self.samples=sum([file[1]-sequence_length for file in self.filenames])
        self.steps_per_epoch=self.samples//self.batchsize

        self.indexes=np.zeros((self.samples,2),dtype=np.int16)
        pos=0
        for i in range(len(self.filenames)):
            for j in range(self.filenames[i][1]-sequence_length):
                self.indexes[pos][0]=i
                self.indexes[pos][1]=j
                pos+=1
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self,idx):
        indices=self.indexes[idx*self.batchsize:(idx+1)*self.batchsize]

        x_data=np.zeros((self.batchsize,sequence_length,datasize),dtype=np.float32)
        y_data=np.zeros((self.batchsize,datasize),dtype=np.float32)

        for i in range(len(indices)):
            #apply random transposition.
            if self.training:
                trans=np.random.randint(data_trans[0],data_trans[1]+1)
                speed=time_trans[np.random.randint(0,len(time_trans))]
            else:
                trans=0
                speed=0
            
            sequence=np.load(self.filenames[indices[i][0]][0])

            for j in range(sequence_length):
                thing=sequence[indices[i][1]+j]
                if thing<125:
                    if 33<=thing<121:
                        x_data[i][j][thing+trans]=1
                    else:
                        x_data[i][j][thing]=1
                else:
                    d=int(round((thing-124)*(1+speed)))
                    d=max(d,1)
                    d=min(d,100)
                    x_data[i][j][d+124]=1

            thing=sequence[indices[i][1]+sequence_length]
            if thing<125:
                if 33<=thing<121:
                    y_data[i][thing+trans]=1
                else:
                    y_data[i][thing]=1
            else:
                d=int(round((thing-124)*(1+speed)))
                d=max(d,1)
                d=min(d,100)
                y_data[i][d+124]=1

        return x_data,y_data
                
def get_data():
    with open('maestro-v2.0.0.json','r') as json_file:
        data=json.load(json_file)

    train_data=[]
    validation_data=[]
    test_data=[]

    path=os.getcwd()

    for piece in data:
        filename=path+'/processed_beats/'+piece['midi_filename'][5:-5]+'.npy'
        duration=np.load(filename).shape[0]
        if piece['split']=='train':
            train_data.append([filename,duration])
        elif piece['split']=='validation':
            validation_data.append([filename,duration])
        elif piece['split']=='test':
            test_data.append([filename,duration])

    return train_data,validation_data,test_data

def process_data():
    files=sorted(glob.glob('midi_files/*.midi'))
    already=sorted(glob.glob('processed_beats/*.npy'))

    if len(already)==len(files):
        print("Got all files.")
        return None

    for file in files:
        name=file[11:-5]
        if ('processed_beats\\'+name+'.npy') in already:
            print("Already got:",file)
            continue
        try:
            midi=MidiFile(file)
            print("Successfully read:",file)
        except:
            print("Error. Can't read:",file)
            continue

        notes=[i for i in midi]
        buffer=0
        pedals=[0,0]
        velocity=200 #purposely silly.
        keyboard=[0 for i in range(88)]
        holder=[]
        
        for msg in notes:
            buffer+=msg.time
            if msg.type=='control_change':
                #pedals.
                if msg.control==64:
                    if msg.value<64 and pedals[0]==1:
                        #sustain off.
                        totalbuff=min(round(100*buffer),100)
                        if totalbuff!=0:
                            holder.append(211+totalbuff)
                        holder.append(88+88+32+1)
                        buffer=0
                        pedals[0]=0
                    elif msg.value>=64 and pedals[0]==0:
                        #sustain on.
                        totalbuff=min(round(100*buffer),100)
                        if totalbuff!=0:
                            holder.append(211+totalbuff)
                        holder.append(88+88+32)
                        buffer=0
                        pedals[0]=1
                elif msg.control==67:
                    if msg.value<64 and pedals[1]==1:
                        #soft off.
                        totalbuff=min(round(100*buffer),100)
                        if totalbuff!=0:
                            holder.append(211+totalbuff)
                        holder.append(88+88+32+3)
                        buffer=0
                        pedals[1]=0
                    elif msg.value>=64 and pedals[1]==0:
                        #soft on.
                        totalbuff=min(round(100*buffer),100)
                        if totalbuff!=0:
                            holder.append(211+totalbuff)
                        holder.append(88+88+32+2)
                        buffer=0
                        pedals[1]=1
            elif msg.type=='note_on':
                totalbuff=min(round(100*buffer),100)
                if totalbuff!=0:
                    holder.append(211+totalbuff)
                if msg.velocity==0 and keyboard[msg.note-21]==1:
                    keyboard[msg.note-21]=0
                    holder.append(88+msg.note-21)
                elif msg.velocity!=0 and keyboard[msg.note-21]==0:
                    keyboard[msg.note-21]=1
                    if velocity!=math.floor(msg.velocity/4):
                        velocity=math.floor(msg.velocity/4)
                        holder.append(88+88+velocity)
                    holder.append(msg.note-21)
                buffer=0

        holder=np.reshape(holder,(len(holder),))
        holder=holder.astype(np.int16)

        np.save(os.getcwd()+'/processed_beats/'+name,holder)

if __name__=='__main__':
    process_data()
