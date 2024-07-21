from tensorflow.keras import backend as K
import tensorflow as tf

def recall(y_true, y_pred, c):
    pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    true_labels = K.cast(K.argmax(y_true, axis=-1), K.floatx())

    tp = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx()))
    fn = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels != c),K.floatx()))
    tn = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels != c),K.floatx()))
    fp = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels == c),K.floatx()))

    sensitivity = (tp)/(tp+fn+0.0000001)
    return sensitivity

def precision(y_true, y_pred, c):
    pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    true_labels = K.cast(K.argmax(y_true, axis=-1), K.floatx())

    tp = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx()))
    fn = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels != c),K.floatx()))
    tn = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels != c),K.floatx()))
    fp = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels == c),K.floatx()))

    precision = (tp)/(tp+fp+0.0000001)
    return precision

def recall_c0(y_true, y_pred):
    return recall(y_true, y_pred, 0)

def precision_c0(y_true, y_pred):
    return precision(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return recall(y_true, y_pred, 1)

def precision_c1(y_true, y_pred):
    return precision(y_true, y_pred, 1)

def recall_c2(y_true, y_pred):
    return recall(y_true, y_pred, 2)

def precision_c2(y_true, y_pred):
    return precision(y_true, y_pred, 2)