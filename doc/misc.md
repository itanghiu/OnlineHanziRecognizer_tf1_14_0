

### TODO:

- replace the dictionary passed in to tf by tf.queue (cf "tf for deep learning ch7)
- add batch normalization
- use tf.getVariables()
- replace Relu by Elu
- replace AdamOptimizer with MomentumOptimizer()
- change the Xavier initializzation of fc layers to He initialization


L1 and L2 

http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

L1 = sum abs(yi-f(xi))
L2 = sum abs( (yi-f(xi)) x (yi-f(xi)))

Intuitively speaking, since a L2-norm squares the error (increasing by a lot if error > 1), the model will see a much larger error ( e vs e^2 ) than the L1-norm, so the model is much more sensitive to this example, and adjusts the model to minimize this error. If this example is an outlier, the model will be adjusted to minimize this single outlier case, at the expense of many other common examples, since the errors of these common examples are small compared to that single outlier case.

### Should I use regularization in conv layers ?

https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/20201
To my understanding it seems controversial to use regularization in the conv layers. We use regularization to prevent overfitting. So that a network doesn't memorize the training set and can't do well when predicting outside of the train data. But in conv layer we don't do any prediction. We actually extract features with these layers. So if we add regularization like dropout we are actually hampering the process of extracting those features effectively. There is no overfitting issue with this layer as it doesn't do prediction so there should be no need for regularization on these layers.


#### L2 regularization :

cf "Fundamentals of deep learning" page 34

### minibatch gradient descent with momentum

cf "Fundamentals of deep learning" page 76
cf " Hand-on machine learning" page 239

### shoudl I use minibatch gradient descent with momentum (optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9))

cf " Hand-on machine learning" page 293 : the conclusion of this section is that you should almost always use
Adam optimization

### learning rate :

decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)

### should I use a initial learning rate =0.01 and halve it every 3 epoch ?

### fc weight initialization :

cf " Hand-on machine learning" page 278 :
By default, the fully_connected() function (introduced in Chapter 10) uses Xavier
initialization (with a uniform distribution). You can change this to He initialization
by using the variance_scaling_initializer() function like this:
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = fully_connected(X, n_hidden1, weights_initializer=he_init, scope="h1")