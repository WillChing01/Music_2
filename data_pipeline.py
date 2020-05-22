from music21 import converter,instrument,note,chord
import numpy as np
import sys
import os
import glob
from keras.utils import Sequence

sys.path.insert(0,os.getcwd())

datasize=88

class Data_Generator(Sequence):

    def __init__(self,filenames,batch_size):
        #filename - [[filename[0],size[0]],[filename[1],size[1]],...]
        self.filenames=filenames[0:len(filenames)-len(filenames)%batch_size]
        self.indexes=np.arange(len(self.filenames))
        self.batch_size=batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.filenames)/self.batch_size)

    def __getitem__(self,idx):
        files=self.filenames[idx*self.batchsize:(idx+1)*self.batchsize]

def process_data():
    files=glob.glob('midi_files/*.midi')

    for file in files:
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

        name=file[11:-5]
        np.save(os.getcwd()+'\\processed_beats\\'+name,holder)

if __name__=='__main__':
    process_data()
