from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import math
import csv
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available())
f = open('angle_test.csv', 'r')
rdr = csv.reader(f)
y_train = []
x_train_list = []
plx = 41.41 / 480
for line in rdr:
    y_train.append(float(line[0]))
    # x_train.append(1/float(line[1]))
    # x_train.append(1/(math.tan(math.radians(float(line[1])*plx))))
    #x_train_list.append(math.radians(float(line[1])*180.0/480.0)+math.pi/2.0)
    # x_train_list.append(float(320-float(line[1])))
    x_train_list.append(float(float(line[1])))
x_train = np.array(x_train_list)
# print(y_train)
# print(x_train)
# x_train = [1,2,3,4]
# y_train = [1,2,3,1]
# T=a^(R)+b

x=[1.0,1.0,1.0,1.0]
#w = tf.Variable(-1.6965117, name='weight')
horizontal_angle_of_view = tf.constant(-53.5, name='a')
b = tf.Variable(-0.43346, name='b') #-0.4311409
#c = tf.Variable(3.8651228, name='c')

degree = tf.constant(90.0, name="degree")
plx_max = tf.constant(320.0, name="plx_max")
#d = tf.Variable(tf.random.normal([1]), name='d')
# hypothesis = -1/(w*(x_train)) - b #x_train * w + b

# 0.14608622

# hypothesis = w * x_train + b

#hypothesis = w * x_train + b

#tf.experimental.numpy.heaviside

#hypothesis =( 180/math.pi)*(tf.atan(plx_max*tf.tan((math.pi/180)*(horizontal_angle_of_view))/x_train))+180.0*np.heaviside(x_train*-1,1.0)+b
hypothesis =tf.atan(x_train*tf.tan(horizontal_angle_of_view*math.pi/360.0)/320.0)*180.0/math.pi+90.0+b



# (tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(x_train)))) * tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(x_train)))))

# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean((hypothesis - y_train)**2)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
while sess.run(cost) > 0.005911372 or np.isnan(sess.run(cost)):
    sess.run(train)
    if step % 100 == 0:

        print(step, "cost: ", sess.run(cost),"horizontal_angle_of_view: ",  sess.run(horizontal_angle_of_view),"b: ",  sess.run(b))#,"d: ", sess.run(d))# sess.run(cost), sess.run(w), sess.run(a), sess.run(b), sess.run(c))
        #print(sess)
        print("=================================================")
    step += 1
    if np.isnan(sess.run(cost)) and np.isnan(sess.run(w)) and step > 100:  # and np.isnan(sess.run(b)):
        print(sess.run(cost),sess.run(w))
        sess.run(tf.global_variables_initializer())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        step = 0

print(step, "cost: ", sess.run(cost),"horizontal_angle_of_view: ",  sess.run(horizontal_angle_of_view))
print("=================================================")
print(x)

