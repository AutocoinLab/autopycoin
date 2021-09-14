"""Checks used to modified inputs."""

import tensorflow as tf


def check_infinity(tensor):
    """Defines checks used to verify infinite values and mask them.

    parameters
    ----------
    tensor : tensor-like.
        Tensor to check.

    returns
    -------
    new_tensor : tensor-like.
        new_tensor masked.
    """

    mask = tf.reduce_all(tf.math.is_finite(tensor), axis=1)
    mask = tf.squeeze(mask)
    new_tensor = tf.boolean_mask(tensor, mask, axis=None, name="boolean_mask")
    new_tensor = tf.expand_dims(new_tensor, axis=1)

    return new_tensor
