import tensorflow as tf 
import numpy
import pandas as pd
#df=pd.read_csv('./Iris_training.csv',usecols = [0,1,2,3,4],skiprows = [0],header=None)
df=pd.read_csv('C:/Users/aroug/Source/Repos/MyTensorFlow/Iris_training.csv',usecols = [0,1,2,3,4],header=0)
d = df.values
print(d)
#l = pd.read_csv('./Iris_training.csv',usecols = [5] ,header=None)
l = pd.read_csv('C:/Users/aroug/Source/Repos/MyTensorFlow/Iris_training.csv',usecols = [5] ,header=-1)
labels = l.values
print(labels)
data = numpy.float32(d)
labels = numpy.array(l,'str')
print(data, labels)

#tensorflow
x = tf.placeholder(tf.float32,shape=(150,5))
x = data
w = tf.random_normal([100,150],mean=0.0, stddev=1.0, dtype=tf.float32)
y = tf.nn.softmax(tf.matmul(w,x))

with tf.Session() as sess:
    print sess.run(y)
