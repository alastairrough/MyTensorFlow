# source https://github.com/sherrym/tf-tutorial/  
import tensorflow as tf
import matplotlib

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))