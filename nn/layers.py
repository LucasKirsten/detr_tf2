import tensorflow as tf

class Parameter(tf.keras.layers.Layer):
    '''
    Analogous to nn.Parameter from PyTorch
    '''
    def __init__(self, shape, **kwargs):
        super(Parameter, self).__init__(**kwargs)
        
        self.shape = shape
        self.param = self.add_weight(name='kernel', 
                                      shape=shape,
                                      initializer='glorot_uniform',
                                      trainable=True)
        
        super(Parameter, self).build(shape)
    
    @tf.function
    def call(self, x):
        return tf.ones([tf.shape(x)[0],
                        *self.param.shape]) * self.param
    
    def get_config(self):
        return {
            'shape' : self.shape
        }