import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.01

mnist=input_data.read_data_sets('./mnist',one_hot=True)
test_x=mnist.test.images[:2000]
test_y=mnist.test.labels[:2000]

tf_x=tf.placeholder(tf.float32,[None,TIME_STEP*INPUT_SIZE])
image=tf.reshape(tf_x,[-1,TIME_STEP,INPUT_SIZE])
tf_y=tf.placeholder(tf.int32,[None,10])

#RNN
run_cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs,(h_c,h_n)=tf.nn.dynamic_rnn(
    run_cell,
    image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False
)
output=tf.layers.dense(outputs[:,-1,:],10)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(LR).minimize(loss)

accuracy=tf.metrics.accuracy(
    labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1),
)[1]

sess=tf.Session()
init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)

for step in range(1200):
    b_x,b_y=mnist.train.next_batch(BATCH_SIZE)
    _,loss_=sess.run([train_op,loss],{tf_x:b_x,tf_y:b_y})
    if step%50==0:
        accuracy_=sess.run(accuracy,{tf_x:test_x,tf_y:test_y})
        print('train loss:%.4f'%loss_,'| test accuracy:%.2f'%accuracy_)

test_output=sess.run(output,{tf_x:test_x[:10]})
pred_y=np.argmax(test_output,1)
print(pred_y,'prediction number')
print(np.argmax(test_y[:10],1),'real number')