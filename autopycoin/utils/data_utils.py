"""Checks used to modified inputs."""

import tensorflow as tf
from tensorflow.keras.backend import floatx


def check_infinity(tensor):
    """
    This funxtion is used to verify infinite values and to mask them.

    Parameters
    ----------
    tensor : tensor-like.
        Tensor to check.

    Returns
    -------
    new_tensor : tensor-like.
        new_tensor masked.
    """

    mask = tf.reduce_all(tf.math.is_finite(tensor), axis=1)
    mask = tf.squeeze(mask)

    if isinstance(tensor, tf.RaggedTensor):
        new_tensor = tf.ragged.boolean_mask(tensor, mask, name="boolean_mask")
    else:
        new_tensor = tf.boolean_mask(tensor, mask, axis=None, name="boolean_mask")

    return new_tensor


def range_dims(tensor, shape):
    """
    Convenient function which performs a range and a reshape operation.

    Parameters
    ----------
    tensor : tensor, array or list
    shape : tuple

    Returns
    -------
    tensor : tensor
    """
    tensor = tf.range(tensor, dtype=floatx())
    tensor = tf.reshape(tensor, shape=shape)
    return tensor
