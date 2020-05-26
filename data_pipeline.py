from mido import MidiFile
import numpy as np
import sys
import os
import glob
import json
import math
from keras.utils import Sequence

sys.path.insert(0,os.getcwd())

datasize=88+88+100 #note_on,note_off,time_shift
sequence_length=64

data_trans=[-3,3]
time_trans=[-0.05,-0.025,0,0.025,0.05]

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

        x_data=np.zeros((self.batchsize,sequence_length,datasize),dtype=np.int8)
        y_data=np.zeros((self.batchsize,datasize),dtype=np.int8)

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
                if thing<176:
                    x_data[i][j][thing+trans]=1
                else:
                    d=round((thing-175)*(1+speed))
                    d=max(d,1)
                    d=min(d,100)
                    x_data[i][j][d+175]=1

            thing=sequence[indices[i][1]+sequence_length]
            if thing<176:
                y_data[i][thing+trans]=1
            else:
                d=round((thing-175)*(1+speed))
                d=max(d,1)
                d=min(d,100)
                y_data[i][d+175]=1

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
        holder=[]
        
        for msg in notes:
            buffer+=msg.time
            if msg.type=='control_change':
                #pedals+sfx
                pass
            elif msg.type=='note_on':
                totalbuff=min(round(100*buffer),100)
                if totalbuff!=0:
                    holder.append(88+88+totalbuff-1)
                if msg.velocity==0:
                    holder.append(88+msg.note-21)
                else:
                    holder.append(msg.note-21)
                buffer=0

        holder=np.reshape(holder,(len(holder),))
        holder=holder.astype(np.int16)

        np.save(os.getcwd()+'/processed_beats/'+name,holder)

if __name__=='__main__':
    process_data()
