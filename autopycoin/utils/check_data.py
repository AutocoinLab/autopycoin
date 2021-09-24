"""Checks used to modified inputs."""

import tensorflow as tf


def check_infinity(tensor):
    """`check_infinity` Defines a control data function. It is
    used to verify infinite values and to mask them.

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

    print("n", tensor)

    if isinstance(tensor, tf.RaggedTensor):
        new_tensor = tf.ragged.boolean_mask(tensor, mask, name="boolean_mask")
    else:
        new_tensor = tf.boolean_mask(tensor, mask, axis=None, name="boolean_mask")

    # new_tensor = tf.expand_dims(new_tensor, axis=1)

    print("t", new_tensor)

    return new_tensor
