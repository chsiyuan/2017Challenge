import tensorflow as tf
import numpy as np
import roi_pooling_op
import roi_pooling_op_grad
import tensorflow as tf
import pdb


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# array = np.random.rand(32, 50, 50, 3)
array = np.zeros((32, 50, 50, 3))
for b in xrange(32):
	for x in xrange(50):
		for y in xrange(50):
			for c in xrange(3):
				array[b,x,y,c] = x+y
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois_array = [[0, 10, 10, 15, 15], [31, 30, 30, 35, 35]]
rois = tf.convert_to_tensor(rois_array, dtype=tf.float32)

# W = weight_variable([3, 3, 3, 1])
W = tf.Variable(tf.convert_to_tensor(np.ones((3, 3, 3, 1)), dtype=tf.float32))
h = conv2d(data, W)
proposal1 = h[0,10:15,10:15,:]
proposal2 = h[31,30:35,30:35,:]
pdb.set_trace()

[y, argmax_x, argmax_y] = roi_pooling_op.roi_pool(h, rois, 3, 3, 1.0)
pdb.set_trace()
y_data = tf.convert_to_tensor(np.ones((2, 3, 3, 2)), dtype=tf.float32)
pdb.set_trace()

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
pdb.set_trace()
for step in xrange(10):
    y_np, loss_np, _ = sess.run([y,loss,train])
    pdb.set_trace()

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
