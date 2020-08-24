import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.utils import Progbar
import tensorflow.keras.layers as KL
from tensorflow.keras import applications as KA

from .transformer import Transformer
from .layers import Parameter

class DETR(tf.keras.models.Model):
    def __init__(self, num_classes, max_detections=100, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, backbone='resnet50'):
        
        super(DETR, self).__init__()
        
        if backbone=='resnet50':
            self.backbone = KA.ResNet50V2(include_top=False, pooling=None)
        elif backbone=='mobilenetv2':
            self.backbone = KA.MobileNetV2(include_top=False, pooling=None, alpha=0.35)
        else:
            raise Exception(f'Backbone {backbone} is currently not available!')
        
        self.transformer = Transformer(num_encoder_layers,num_decoder_layers, \
                                       d_model=hidden_dim, num_heads=nheads, dff=hidden_dim,
                                       pe_input=1000, pe_target=max_detections)
        
        self.conv = KL.Conv2D(hidden_dim, (1,1), activation='relu')
        self.flatten = KL.Reshape((-1, hidden_dim))
        
        # output positional encodings (object queries)
        self.query_pos = Parameter((max_detections, hidden_dim))
        
        # prediction heads, one extra class for predicting non-empty slots
        self.linear_class = KL.Dense(num_classes+1, activation='softmax')
        self.linear_bbox  = KL.Dense(4, activation='sigmoid')
        
        self.concatenate = KL.Concatenate(axis=-1)
        
    def call(self, x, training=False):
        
        x = self.backbone(x)
        x = self.conv(x)
        x = self.flatten(x)
        
        q = self.query_pos(x)
        h = self.transformer(x, q, training)
        
        classes = self.linear_class(h)
        boxes   = self.linear_bbox(h)
        
        return classes, boxes
    
    def fit(self,
            x=None,
            epochs=1,
            callbacks=None,
            validation_data=None,
            steps_per_epoch=None,
            validation_steps=None,
            *args, **kwargs):
        
        total_epochs = epochs
        train_generator = x
        val_generator = validation_data
        callback_list = CallbackList(callbacks=callbacks, model=self)

        self.stop_training = False
        callback_list.on_train_begin()
        for epoch in range(total_epochs):
            
            # logs for callbacks
            logs = {}
            
            ''' Train step '''
            print(f'\nEpoch {epoch+1}/{total_epochs}')
            train_loss = {'class_loss':[], 'boxes_loss':[], 'loss':[]}
            pbar = Progbar(steps_per_epoch)
            for it in range(steps_per_epoch):
                batchX, batchY = next(train_generator)
                
                with tf.GradientTape() as tape:
                    batchPred = self(batchX, training=True)
                    
                    # permute the indices of y_true to match y_pred
                    with tape.stop_recording():
                        batchY = self.loss.permute_indx(batchY, batchPred)
                    
                    # calculate loss
                    losses = self.loss(*batchY, *batchPred)
                    loss = tf.reduce_sum(losses, axis=0)
                    
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients((grad, var) 
                                                for (grad, var) in zip(gradients, self.trainable_variables) 
                                                if grad is not None)

                train_loss['class_loss'].append(np.mean(losses[0]))
                train_loss['boxes_loss'].append(np.mean(losses[1]))
                train_loss['loss'].append(np.mean(np.concatenate(losses, axis=-1)))
                pbar.update(current=it, values=[(key, train_loss[key][-1]) for key in train_loss.keys()])
                
            logs['loss'] = np.mean(train_loss['loss'])

            ''' Validation step '''
            if validation_data is not None:
            
                print(f'\nValidation {epoch+1}/{total_epochs}')
                val_loss = {'val_class_loss':[], 'val_boxes_loss':[], 'val_loss':[]}
                pbar = Progbar(validation_steps)
                for it in range(validation_steps):
                    batchX, batchY = next(val_generator)
                    batchPred = self(batchX, training=False)

                    batchY = self.loss.permute_indx(batchY, batchPred)
                    losses = self.loss(*batchY, *batchPred)

                    val_loss['val_class_loss'].append(np.mean(losses[0]))
                    val_loss['val_boxes_loss'].append(np.mean(losses[1]))
                    val_loss['val_loss'].append(np.mean(np.concatenate(losses, axis=-1)))
                    pbar.update(current=it, values=[(key, val_loss[key][-1]) for key in val_loss.keys()])
                    
                    logs['val_loss'] = np.mean(val_loss['val_loss'])
                    
            callback_list.on_epoch_end(epoch+1, logs=logs)
            
            print()
            # if any callback stop training
            if self.stop_training:
                callback_list.on_train_end(logs={'loss':np.mean(train_loss['loss']),
                                                 'val_loss':np.mean(val_loss['val_loss'])})
                break