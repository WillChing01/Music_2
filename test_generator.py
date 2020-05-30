import sys
import os
import glob
import numpy as np
import tensorflow as tf
import time

sys.path.insert(0,os.getcwd())

from data_pipeline import *

class Data_Generator2(object):

    def __init__(self,filenames):
        self.filenames=sorted(filenames,key=lambda x:x[1],reverse=True)
        self.samples=sum([file[1]-sequence_length for file in self.filenames])

        self.indexes=np.zeros((self.samples,2),dtype=np.int32)
        pos=0
        for i in range(len(self.filenames)):
            for j in range(self.filenames[i][1]-sequence_length):
                self.indexes[pos][0]=i
                self.indexes[pos][1]=j
                pos+=1

    def get_next_data(self):
        while True:
            index=np.random.randint(0,self.samples)

            yield (self.filenames[self.indexes[index][0]][0],
                   self.indexes[index][1])

class Dataset(object):

    def __init__(self,filenames):
        generator=Data_Generator2(filenames)
        self.iterator=self.build_iterator(generator)

    def build_iterator(self,gen:Data_Generator2):
        batch_size=64
        prefetch_batch_buffer=10
        parallel=4

        self.dataset=tf.data.Dataset.from_generator(gen.get_next_data,
                                                    output_shapes=(tf.TensorShape([]),
                                                                   tf.TensorShape([])),
                                                    output_types=(tf.string,
                                                                  tf.int32))

        self.dataset=self.dataset.map(lambda x,y:tf.py_function(self.tidydata,[x,y],[tf.float32,tf.float32]),
                                      num_parallel_calls=parallel)
        self.dataset=self.dataset.batch(batch_size)
        self.dataset=self.dataset.prefetch(prefetch_batch_buffer)
        i=self.dataset.as_numpy_iterator()
        return i

    def tidydata(self,filename,index):
        #read files from disk.
        filename=filename.numpy().decode()
        index=index.numpy()
        sequence=np.load(filename)

        trans=np.random.randint(data_trans[0],data_trans[1]+1)
        speed=1+time_trans[np.random.randint(0,len(time_trans))]

        x_data=np.zeros((sequence_length,datasize),dtype=np.float32)
        y_data=np.zeros((datasize),dtype=np.float32)

        for i in range(index,index+sequence_length):
            a=sequence[i]
            if a<125:
                if 33<=a<121:
                    x_data[i-index][a+trans]=1
                else:
                    x_data[i-index][a]=1
            else:
                d=int(round((a-124)*speed))
                d=max(d,1)
                d=min(d,100)
                x_data[i-index][d+124]=1

        a=sequence[index+sequence_length]
        if a<125:
            if 33<=a<121:
                y_data[a+trans]=1
            else:
                y_data[a]=1
        else:
            d=int(round((a-124)*speed))
            d=max(d,1)
            d=min(d,100)
            y_data[d+124]=1

        return (x_data,y_data)

a,b,c=get_data()
dataset=Dataset(b)
start=time.time()
for i in range(100):
    out=next(dataset.iterator)
end=time.time()
print(end-start)
