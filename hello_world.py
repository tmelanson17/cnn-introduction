import tensorflow as tf
import numpy as np

two = tf.constant(2)
add = two + two
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print 'Outputting 2 + 2'
print sess.run(add)

