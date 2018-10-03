

https://github.com/soloice/Chinese-Character-Recognition

#### Start tensorBoard:

>tensorboard --logdir=./log/ --port=8090 --host=127.0.0.1


#### Generate the training and test dataset:

> python ImageDatasetGeneration.py

#### TRAINING data:
Size: 4.42G
Size on disk : 6.23 Go
895 000 files 3 755 folders

#### TEST data:
Size: 1.12G
Size on disk : 1.57 Go
223 991 files 3 755 folders

No validation dataset

An epoch is composed of 895000/100= 8 950 batch


Dropout :

Dropout only applies to the fully-connected region of the network. For all other regions you should not use dropout.
Instead you should insert batch normalization between your convolutions. This will regularize your model, as well as make your model more stable during training.


------------------------------------------

#### Hyperparameters 

##### filter size : 3x3
 
##### stride size : 1

##### zero padding size: 1

##### Activation function: Relu 

##### max pooling : 2x2 window   stride:2

##### batch size: 100

##### momentum: ?

##### Regularization: 
dropout rate: 0.8

##### Evaluation step: every 100 

##### Learning rate: initial: 0.01 . Halved every 3 epoch

##### Number of epoch: 
an epoch is composed of 895 000/100 =  8 950 batch
one step corresponds to one batch
nbr max step = 100 001. Number of epoch = 100 001 batch /8 950 batch = 11

##### initialization: Normal distribution with zero mean and a variance : 0.01 with biases initalized at 0

---------------------------- -----------
#### Hyperparameters of ZHUANG papers

##### filter size : 3
 
##### stride size : 1

##### zero padding size: 1

##### Activation function: Relu 

##### max pooling : 2x2 window   stride:2

##### batch size: 128

##### momentum: 0.9

##### Regularization: weight decay (L2 regularization)  + dropout 
 weight decay: 0.0005 
 dropout rate: 0.5

##### Evaluation: every 0.5 epoch

##### Learning rate: initial: 0.01 . Halved every 3 epoch

##### Number of epoch: 15

##### initialization: Normal distribution with zero mean and a variance : 0.01 with biases initalized at 0

