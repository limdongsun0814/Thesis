import tensorflow as tf
import numpy as np
import math
import csv
import os
import copy
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available())
f = open('distance2.csv', 'r')
rdr = csv.reader(f)
y_train = []
x_train_list = []
x_train_list_all=[]
y_train_all=[]
z_train_list_all,z_train_list=[],[]
plx = 41.41 / 480
for line in rdr:
    if float(line[0]) < 200:
        y_train.append(float(line[0]))
    # x_train.append(1/float(line[1]))
    # x_train.append(1/(math.tan(math.radians(float(line[1])*plx))))
    #x_train_list.append(math.radians(float(line[1])*180.0/480.0)+math.pi/2.0)
        x_train_list.append(float(line[1]))
        z_train_list.append(float(line[2]))
    x_train_list_all.append(float(line[1]))
    y_train_all.append(float(line[0]))
    z_train_list_all.append(float(line[2]))
x_train_all = np.array(x_train_list_all)
x_train = np.array(x_train_list)
z_train_np=np.array(z_train_list)
z_train_np_all=np.array(z_train_list_all)
# print(y_train)
# print(x_train)
# x_train = [1,2,3,4]
# y_train = [1,2,3,1]
# T=a^(R)+b
y_train_array=np.array(y_train)
x=[1.0,1.0,1.0,1.0]#constant
w = tf.Variable(10.6, name='weight')#12.929554,
a = tf.Variable(186.0,name='a')
    #tf.random.normal([1]), name='a')
b = tf.constant(41.41,name='b')#
    #tf.random.normal([1]), name='b')
c = tf.constant(1.0,name='c')
    #tf.random.normal([1]), name='c')
d = tf.Variable(230.96846,name='d')#
    #tf.constant.normal([1]), name='d')
# hypothesis = -1/(w*(x_train)) - b #x_train * w + b

# 0.14608622

# hypothesis = w * x_train + b

#hypothesis = w * x_train + b
#ATAN(TAN(A)*X/480)

#hypothesis = (w*x_train/(tf.tan(x_train/a+1/b)))+c
#hypothesis = b+w/(tf.tan(tf.atan(a/x_train)*c+d+(x_train*(48.8/480)*(math.pi/180)))+tf.tan(tf.atan(a/x_train)*c+0*d))
#hypothesis = b+abs(w)/(tf.tan(((d-(x_train+z_train_np)*a)*(48.8/480))*(math.pi/180)+(z_train_np*(48.8/480))*(math.pi/180))+tf.tan((d-(x_train+z_train_np)*a)*(48.8/480)*(math.pi/180)))
#hypothesis = b+abs(w)/(tf.tan(((d-x_train)*-48.8/480)*(math.pi/180)+(z_train_np*(48.8/480))*(math.pi/180))+tf.tan(((d-x_train)*48.8/480)*(math.pi/180)))
hypothesis = w/(tf.tan(tf.atan((z_train_np)*tf.tan(b*math.pi/180)/480)-tf.atan((a-x_train)*tf.tan(b*math.pi/180)/480))+tf.tan(tf.atan((a-x_train)*tf.tan(b*math.pi/180)/480)))

#hypothesis = w/(tf.tan(x_train*a+d+(x_train*(48.8/480)*(math.pi/180)))+tf.tan(x_train*a+d))
# (tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(x_train)))) * tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(tf.math.sqrt(x_train)))))

cost = tf.reduce_mean((hypothesis - y_train)**2)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
old_cost = 0
aa,bb,cc,dd,ww=0,0,0,0,0
while sess.run(cost) > 0.05911372 or np.isnan(sess.run(cost)):
    sess.run(train)
    if step % 100 == 0:

        print(step, "cost: ", sess.run(cost),"w: ", sess.run(w),"a: ",  sess.run(a),"b: ", sess.run(b),"c: ", sess.run(c),"d: ", sess.run(d))#,"d: ", sess.run(d))# sess.run(cost), sess.run(w), sess.run(a), sess.run(b), sess.run(c))
        #print(sess)
        #hypothesis_np=sess.numpy()
        aa=sess.run(a)
        bb=sess.run(b)
        cc=sess.run(c)
        dd=sess.run(d)
        ww=sess.run(w)
        #val = bb + abs(ww) / (np.tan(((dd - x_train_all) * -48.8 / 480) * math.pi / 180 + (
        #        z_train_np_all * (48.8 / 480)) * math.pi / 180) + np.tan(
        #    ((dd - x_train_all) * 48.8 / 480) * math.pi / 180))
        val = ww/(np.tan(np.arctan((z_train_np_all)*np.tan(bb*math.pi/180)/480)-np.arctan((aa-x_train_all)*np.tan(bb*math.pi/180)/480))+np.tan(np.arctan((aa-x_train_all)*np.tan(bb*math.pi/180)/480)))

        #val = bb+abs(ww)/(np.tan((dd-(x_train_all+z_train_np_all)*aa)*(48.8/480)*math.pi/180+(z_train_np_all*(48.8/480))*math.pi/180)+np.tan((dd-(x_train_all+z_train_np_all)*aa)*(48.8/480)*math.pi/180))
        plt.clf()
        plt.plot(val-y_train_all,'*')
        #plt.ylim(-20,20)
        plt.show(block=False)
        plt.pause(0.0000000000000000000000000001)
        print("=================================================")


        #if old_cost == sess.run(cost):
            #sess.run(tf.global_variables_initializer())
            #step = 0
            #old_cost=0
        #else:
            #old_cost=sess.run(cost)
    step += 1
    if np.isnan(sess.run(cost)) and np.isnan(sess.run(w)) and step > 100:  # and np.isnan(sess.run(b)):
        print(sess.run(cost),sess.run(w))
        sess.run(tf.global_variables_initializer())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        step = 0

print("cost: ", sess.run(cost),"w: ", sess.run(w),"a: ",  sess.run(a),"b: ", sess.run(b),"c: ", sess.run(c),"d: ", sess.run(d))#,"d: ", sess.run(d), )  # "b: ",sess.run(b),
print("=================================================")
print(x)

