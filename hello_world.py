import tensorflow as tf
import numpy as np

two = tf.constant(2)
add = two + two
sess = tf.Session()
print 'Outputting 2 + 2'
print sess.run(add)

