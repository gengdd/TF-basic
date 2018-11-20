import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.1,size=x.shape)
y=np.power(x,2)+noise

# print(x.shape)
# print(y.shape)
# plt.scatter(x,y)
# plt.show()

tf_x=tf.placeholder(tf.float32,x.shape)
tf_y=tf.placeholder(tf.float32,y.shape)

# l1=tf.layers.dense(tf_x,10,tf.nn.relu)
# output=tf.layers.dense(l1,1)

# loss=tf.losses.mean_squared_error(tf_y,output)
# optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
# train_op=optimizer=optimizer.minimize(loss)

# sess=tf.Session()
# sess.run(tf.global_variables_initializer())

# plt.ion()

# for step in range(100):
#     _,l,pred=sess.run([train_op,loss,output],feed_dict={tf_x:x,tf_y:y})
#     if step%5==0:
#         print(l)
#         plt.cla()
#         plt.scatter(x,y)
#         plt.plot(x,pred,'r-',lw=5)
#         plt.text(0.5,0,'Loss=%.4f'%l,fontdict={'size':20,'color':'red'})
#         plt.pause(0.1)

# plt.ioff()
# plt.show()

# w1=tf.Variable(tf.random_normal([1,10]))
# b1=tf.Variable(tf.zeros([1,10]))
# wxb1=tf.matmul(tf_x,w1)+b1
# L1=tf.nn.tanh (wxb1)

# w2=tf.Variable(tf.random_normal([10,1]))
# b2=tf.Variable(tf.zeros([1]))
# wxb2=tf.matmul(L1,w2)+b2
# prediction=tf.nn.tanh(wxb2)

# loss=tf.reduce_mean(tf.square(tf_y-prediction))
# optimizer=tf.train.GradientDescentOptimizer(0.5)
# train_op=optimizer.minimize(loss)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    
#     plt.ion()

#     for step in range(1000):
#         _,l,pred=sess.run([train_op,loss,prediction],feed_dict={tf_x:x,tf_y:y})
#         if step%5==0:
#             print(l)
#             plt.cla()
#             plt.scatter(x,y)
#             plt.plot(x,pred,'r-',lw=5)
#             plt.text(0.5,0,'Loss=%.4f'%l,fontdict={'size':20,'color':'red'})
#             plt.pause(0.1)

# plt.ioff()
# plt.show()

