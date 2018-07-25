# coding=utf-8
 
import tensorflow as tf
import numpy as np
 
def py_func(func, inp, Tout, stateful = True, name=None, grad_func=None):  
    rand_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))  
    tf.RegisterGradient(rand_name)(grad_func)  
    g = tf.get_default_graph()  
    with g.gradient_override_map({'PyFunc':rand_name}):  
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)  
  
def coco_forward(xw, y, m, name=None):  
    #pdb.set_trace()  
    xw_copy = xw.copy()  
    num = len(y)  
    orig_ind = range(num)  
    xw_copy[orig_ind,y] -= m  
    return xw_copy  
  
def coco_help(grad,y):  
    grad_copy = grad.copy()  
    return grad_copy  
  
def coco_backward(op, grad):  
      
    y = op.inputs[1]  
    m = op.inputs[2]  
    grad_copy = tf.py_func(coco_help,[grad,y],tf.float32)  
    return grad_copy,y,m  
  
def coco_func(xw,y,m, name=None):  
    with tf.op_scope([xw,y,m],name,"Coco_func") as name:  
        coco_out = py_func(coco_forward,[xw,y,m],tf.float32,name=name,grad_func=coco_backward)  
        return coco_out  
  
def cos_loss(x, y,  num_cls, reuse=False, alpha=0.25, scale=64,name = 'cos_loss'):  
    ''''' 
    x: B x D - features 
    y: B x 1 - labels 
    num_cls: 1 - total class number 
    alpah: 1 - margin 
    scale: 1 - scaling paramter 
    '''  
    # define the classifier weights  
    xs = x.get_shape()  
    y = tf.reshape(tf.cast(y, dtype = tf.int32),[-1])
    with tf.variable_scope('centers_var',reuse=reuse) as center_scope:  
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32,   
            initializer=tf.contrib.layers.xavier_initializer(),trainable=True)  
     
    #normalize the feature and weight  
    #(N,D)  
    x_feat_norm = tf.nn.l2_normalize(x,1,1e-10)  
    #(D,C)  
    w_feat_norm = tf.nn.l2_normalize(w,0,1e-10)  
      
    # get the scores after normalization   
    #(N,C)  
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)    
    #value = tf.identity(xw)  
    #substract the marigin and scale it  
    value = coco_func(xw_norm,y,alpha) * scale  
      
    # compute the loss as softmax loss  
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))  
  
    return cos_loss
