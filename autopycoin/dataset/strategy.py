"""
Defines strategy functions.
"""

#TODO: Unit testing
import tensorflow as tf


def features(inputs, features_slice, columns_index):
    feature_length = features_slice.stop - features_slice.start
    feature = tf.stack([inputs[..., features_slice, index] for index in columns_index], axis=-1)
    feature.set_shape([None, feature_length, len(columns_index)])
    return feature

def date_features(inputs, features_slice, columns_index) -> tf.Tensor:
    """Return input and output date tensors from the features tensor."""
    date = features(inputs, features_slice, columns_index)
    date = tf.cast(date, tf.int32)
    date = tf.strings.as_string(date)
    return tf.strings.reduce_join(date, separator="-", axis=-1, keepdims=True)
