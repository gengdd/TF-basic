import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

npx=np.random.uniform(-1,1,(1000,1))
npy=np.power(npx,2)+np.random.normal(0,0.1,size=npx.shape)
npx_train,npx_test=np.split(npx,[800])
npy_train,npy_test=np.split(npy,[800])

# plt.scatter(npx_test,npy_test)
# plt.show()

tfx=tf.placeholder(npx_train.dtype,npx_train.shape)
tfy=tf.placeholder(npy_train.dtype,npy_train.shape)

dataset=tf.data.Dataset.from_tensor_slices((tfx,tfy))
dataset=dataset.shuffle(buffer_size=1000)
dataset=dataset.batch(32)
dataset=dataset.repeat(3)
iterator=dataset.make_initializable_iterator()

bx,by=iterator.get_next()
l1=tf.layers.dense(bx,10,tf.nn.relu)
out=tf.layers.dense(l1,npy.shape[1])
loss=tf.losses.mean_squared_error(by,out)
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess=tf.Session()
sess.run([iterator.initializer,tf.global_variables_initializer()],feed_dict={tfx:npx_train,tfy:npy_train})

for step in range(201):
    try:
        _,train1=sess.run([train,loss])
        if step%10==0:
            test1=sess.run(loss,{bx:npx_test,by:npy_test})
            print('step:%i/200'%step,'|train loss:',train1,'|test loss:',test1)
    except tf.errors.OutOfRangeError:
        print('Finish the last epoch')
        break
