import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import numpy as np
from scipy.optimize import linear_sum_assignment

def cxcy_to_xy(boxes):
    # get boxes in cx,cy,w,h
    cx = boxes[...,0]
    cy = boxes[...,1]
    w  = boxes[...,2]
    h  = boxes[...,3]
    
    # convert to xmin, ymin, xmax, ymax
    xmin = cx - w/2
    ymin = cy - h/2
    xmax = cx + w/2
    ymax = cy + h/2
    
    boxes =  tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return tf.clip_by_value(boxes, 0, 1)

def intersection_over_union(boxA, boxB):
    max = lambda x:tf.reduce_max(tf.stack(x, axis=-1), axis=-1)
    min = lambda x:tf.reduce_min(tf.stack(x, axis=-1), axis=-1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max([boxA[...,0], boxB[...,0]])
    yA = max([boxA[...,1], boxB[...,1]])
    xB = min([boxA[...,2], boxB[...,2]])
    yB = min([boxA[...,3], boxB[...,3]])

    # compute the area of intersection rectangle
    interArea = max([xB - xA + 1, tf.zeros_like(xA)]) * max([yB - yA + 1, tf.zeros_like(xA)])

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[...,2] - boxA[...,0] + 1) * (boxA[...,3] - boxA[...,1] + 1)
    boxBArea = (boxB[...,2] - boxB[...,0] + 1) * (boxB[...,3] - boxB[...,1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return 1-iou

def weighted_class_loss(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * (1.-y_pred) * weights
        #loss = K.sum(loss, -1)
        return loss
    
    return loss

class MatchLoss():
    
    def __init__(self, num_classes, w_l1=1., w_iou=1., w_noobject=1/2.):
        self.num_classes = num_classes
        self.w_l1  = w_l1
        self.w_iou = w_iou
        self.w_noobject = w_noobject
        
        # categorical crossentropy loss
        weights = [1.]*(num_classes+1)
        weights[0] = w_noobject
        self.crossentropy = weighted_class_loss(weights)

        # mean absolute error
        self.l1 = lambda true, pred : tf.abs(true-pred)

        # GIoU
        self.iou_loss = tfa.losses.GIoULoss() # can also use intersection_over_union function

    @tf.function
    def __call__(self, true_class, true_boxes, pred_class, pred_boxes, single=False):
        '''
        Call on loss function.
        '''

        # convert from cx,cy,w,h to xmin,ymin,xmax,ymax
        xy_true_boxes = cxcy_to_xy(true_boxes)
        xy_pred_boxes = cxcy_to_xy(pred_boxes)

        # this is the value for having or not objects
        # so it can calculate or no the losses for boxes
        has_object = 1. - true_class[...,0]
        
        # calculate the 3 losses
        class_loss = self.crossentropy(true_class, pred_class)
        boxes_l1  = tf.stack([has_object]*4, axis=-1) * self.w_l1 * self.l1(true_boxes, pred_boxes)
        boxes_iou = has_object * self.w_iou * self.iou_loss(xy_true_boxes, xy_pred_boxes)
        
        # normalize boxes
        if single:
            total_boxes = tf.math.count_nonzero(has_object, dtype=tf.float32) + K.epsilon()
            
        else:
            # get the number of boxes per batch
            total_boxes = tf.map_fn(
                fn = lambda x:tf.math.count_nonzero(x, dtype=tf.float32) + K.epsilon(),
                elems = has_object[:,0,:],
            )
            # reshape to match the boxes_loss shape
            total_boxes = tf.stack([[total_boxes]*boxes_l1.shape[-2]]*boxes_l1.shape[-3], axis=-1)
            total_boxes = K.permute_dimensions(total_boxes, (1,2,0))
        
        return K.sum(class_loss, axis=-1), (K.sum(boxes_l1, axis=-1)+boxes_iou)/total_boxes
    
    def permute_indx(self, y_true, y_pred):
        '''
        Permute the indexes of y_true to match the smaller loss
        '''
        
        # size of the input batch
        batch_size = tf.shape(y_true[0])[0]
        
        # hungarian calc for permutation indices
        def hungarian(true, pred):
            
            # calculate cost matrix
            costs = np.zeros([true[0].shape[0], true[0].shape[0]])
            for i in tf.range(tf.shape(costs)[0]):
                for j in tf.range(tf.shape(costs)[0]):
                    costs[i][j] = tf.reduce_sum(
                                    self.__call__(true[0][i], true[1][i], pred[0][j], pred[1][j], single=True)
                                )
            
            # look for lower cost assignment
            _, col_ind = linear_sum_assignment(costs)
            
            # permute elements of ground truth
            true_class = tf.gather(true[0], col_ind)
            true_boxes = tf.gather(true[1], col_ind)

            # calculate loss on permuted elements
            return true_class, true_boxes
        
        true_class = []; true_boxes = []
        for b in tf.range(batch_size):
            tc, tb = hungarian((y_true[0][b], y_true[1][b]),\
                               (y_pred[0][b], y_pred[1][b]))
            true_class.append(tc)
            true_boxes.append(tb)
        
        return tf.stack(true_class, axis=0), tf.stack(true_boxes, axis=0)