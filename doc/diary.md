
# DIARY

#2020-11-20
premiere execution de hanzirecog tf2 avec loss="categorical_crossentropy"
La valeur du loss est bien trop elevé: 188 000 au debut et ne baisse pas apres.
Je vais essayer de remplacer le loss par le custom loss utilisé dans hazirecog tf1.
cf: hands-on with sci-kit chap 12 custom loss function


## 2020-10-09
PROBLEM: After having converted Data.py to tf2 using the tf_upgrade_v2 script I got :
AttributeError: 'BatchDataset' object has no attribute 'output_types'
for the line :
dataset = tf.data.Dataset.zip((image_file_path_dataset, label_dataset)).shuffle(500).repeat().batch(self.batch_size_ph)
self.iterator = tf.compat.v1.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
SOLUTION:
Replaced :
dataset = tf.data.Dataset.zip((image_file_path_dataset, label_dataset)).shuffle(500).repeat().batch(self.batch_size_ph)
with:
dataset = tf.compat.v1.data.Dataset.zip((image_file_path_dataset, label_dataset)).shuffle(500).repeat().batch(self.batch_size_ph)


## 2020-09-09
PROBLEM: I want to upgrade hanzirecog to tf2.3. However, the following package :
import tensorflow.contrib.slim as slim
is not recognized
SOLUTON: installed the module tf_slim, with :
> pip install --upgrade tf-slim


## 2020-08-18
Trying to name the tensors predicted_probability_op and predicted_index_op with the statement : tf.identity(my_tensor, name="tensor_name"), does not work.
The only way to be able to retrieve the tensors is to know the real names ('TopKV2:0' and 'TopKV2:1') of the tensors using a debugger. 
The resulting code to retrieve them is :
 predicted_probability_op = graph.get_tensor_by_name('TopKV2:0')
 predicted_index_op = graph.get_tensor_by_name('TopKV2:1')
 
 

 
