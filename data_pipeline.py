from music21 import converter,instrument,note,chord
import numpy as np
import sys
import os
import glob
import json
import math
from keras.utils import Sequence

sys.path.insert(0,os.getcwd())

datasize=88
sequence_length=64
data_trans=[-3,3]

class Data_Generator(Sequence):

    def __init__(self,filenames,batch_size):
        #filename - [[filename[0],size[0]],[filename[1],size[1]],...]
        self.filenames=sorted(filenames,key=lambda x:x[1],reverse=True)
        self.batchsize=batch_size

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
            sequence=np.load(self.filenames[indices[i][0]][0])
            sequence_x=sequence[indices[i][1]:indices[i][1]+sequence_length]
            sequence_y=sequence[indices[i][1]+sequence_length]

            #apply random transposition.
            trans=np.random.randint(data_trans[0],data_trans[1]+1)
            x_data[i]=[np.roll(a,trans,axis=0) for a in sequence_x]
            y_data[i]=np.roll(sequence_y,trans,axis=0)

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

    for file in files:
        name=file[11:-5]
        if ('processed_beats/'+name+'.npy') in already:
            print("Already got file.")
            continue
        try:
            midi=converter.parse(file)
            print("Successfully read:",file)
        except:
            print("Error. Can't read:",file)
            continue
        notes_to_parse=None

        parts=instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse=parts.parts[0].recurse()
        else:
            notes_to_parse=midi.flat.notes

        total_length=0
        for element in notes_to_parse:
            if isinstance(element,note.Note) or isinstance(element,chord.Chord) or isinstance(element,note.Rest):
                if round(4*element.offset)+round(4*element.quarterLength)>total_length:
                    total_length=round(4*element.offset)+round(4*element.quarterLength)

        print(total_length)
        holder=np.zeros((total_length,datasize),dtype=np.int8)
        for element in notes_to_parse:
            if isinstance(element,note.Note) or isinstance(element,chord.Chord):
                offset=round(4*element.offset)
                duration=round(4*element.quarterLength)
                if isinstance(element,chord.Chord):
                    notes=[pitch.midi-21 for pitch in element.pitches]
                    for i in range(duration):
                        for pitch in notes:
                            holder[offset+i][pitch]=1
                else:
                    for i in range(duration):
                        holder[offset+i][element.pitch.midi-21]=1

        np.save(os.getcwd()+'/processed_beats/'+name,holder)

##if __name__=='__main__':
##    process_data()
