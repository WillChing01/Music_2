import numpy as np
import sys
import os
import random
import mido

sys.path.insert(0,os.getcwd())

from data_pipeline import *
from model_architecture import *

def create_song(total_length):
    #creates a random song which is 'total_length' seconds long.
    files=sorted(glob.glob('processed_beats/*.npy'))
    file=files[np.random.randint(0,len(files))]

    piece=np.load(file)
    pattern=[[0 for i in range(datasize)] for x in range(sequence_length)]
    for i in range(sequence_length):
        pattern[i][piece[i]]=1

    total_output=[]

    t=0
    while t<total_length:
        network_input=np.reshape(pattern,(1,sequence_length,datasize))

        output=model.predict(network_input,verbose=0)[0]

        index=np.random.choice(datasize,1,p=output)[0]
        #index=np.argmax(output)

        total_output.append(index)

        new=[0 for i in range(datasize)]
        new[index]=1
        pattern.append(new)
        pattern=pattern[1:len(pattern)]

        if index>211:
            t+=(index-211)/100
            print(t)

    d=0
    midi=mido.MidiFile()
    track=mido.MidiTrack()
    midi.tracks.append(track)

    keyboard=[0 for i in range(88)]
    pedals=[0,0]
    vel=0

    for element in total_output:
        if element>211:
            d+=(element-211)*9.6)
        else:
            if element<88 and keyboard[element]==0:
                #note on.
                keyboard[element]=1
                midi.tracks[0].append(mido.Message('note_on',note=element+21,velocity=vel,time=round(d)))
                d=0
            elif 88<=element<88+88 and keyboard[element-88]==1:
                #note off.
                keyboard[element-88]=0
                midi.tracks[0].append(mido.Message('note_on',note=element-88+21,velocity=0,time=round(d)))
                d=0
            elif 88+88<=element<88+88+32:
                #vel.
                vel=(element-88-88)*4+2
            else:
                if element==88+88+32 and pedal[0]==0:
                    pedal[0]=1
                    midi.tracks[0].append(mido.Message('control_change',control=64,value=127,time=round(d)))
                    d=0
                elif element==88+88+32+1 and pedal[0]==1:
                    pedal[0]=0
                    midi.tracks[0].append(mido.Message('control_change',control=64,value=0,time=round(d)))
                    d=0
                elif element==88+88+32+2 and pedal[1]==0:
                    pedal[1]=1
                    midi.tracks[0].append(mido.Message('control_change',control=67,value=127,time=round(d)))
                    d=0
                elif element==88+88+32+3 and pedal[1]==1:
                    pedal[1]=0
                    midi.tracks[0].append(mido.Message('control_change',control=67,value=0,time=round(d)))
                    d=0

    return midi

try:
    a=model.predict(np.zeros((1,sequence_length,datasize)))
    model.load_weights('weights.hdf5')
    print("Successfully loaded previous weights.")
except:
    print("Cannot load weights. Generating new model.")
