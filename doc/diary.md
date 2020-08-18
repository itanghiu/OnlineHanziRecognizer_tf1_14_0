
# DIARY
## 2020-08-18
Trying to name the tensors predicted_probability_op and predicted_index_op with the statement : tf.identity(my_tensor, name="tensor_name"), does not work.
The only way to be able to retrieve the tensors is to know the real names ('TopKV2:0' and 'TopKV2:1') of the tensors using a debugger. 
The resulting code to retrieve them is :
 predicted_probability_op = graph.get_tensor_by_name('TopKV2:0')
 predicted_index_op = graph.get_tensor_by_name('TopKV2:1')
 
 

 
