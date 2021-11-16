"""Checks used to modified inputs."""

from typing import Union, Tuple, List
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx


def check_infinity(tensor: tf.Tensor) -> tf.Tensor:
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


def range_dims(tensor: tf.Tensor, shape: Tuple, dtype: str = floatx()) -> tf.Tensor:
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
    tensor = tf.range(tensor, dtype=dtype)
    tensor = tf.reshape(tensor, shape=shape)
    return tensor


# corriger
def quantiles_handler(quantiles: List[Union[int, float]]) -> tf.Tensor:
    """
    Convenient function which ensures that quantiles contains the 0.5 quantile
    and is symetric around 0.5. You can feed this function with an int or a float list.
    Negative quantiles are not allowed.

    Parameters
    ----------
    quantiles : list[int or float]
        List of quantiles.

    Returns
    -------
    quantiles : list[float]

    Raises
    ------
    ValueError
        If a quantile is negative then ValueError is raised.
    """

    quantiles = np.append(quantiles, 0.5)
    # *100 in order to suppress float approximation
    quantiles = np.where(quantiles < 1, quantiles * 100, quantiles)

    # symmetry
    quantiles = np.concatenate([quantiles, 100 - quantiles], axis=0)
    quantiles = np.sort(quantiles)
    quantiles = np.unique(quantiles)
    return (quantiles / 100).tolist()


def example_handler(dataset):
    """
    Convenient function which extract one instance of a
    `WindowGenerator` train, validation, test or forecast dataset.

    Parameters
    ----------
    dataset : `WindowGenerator datasets` or tuple[`tensor`]
        `WindowGenerator` datasets train, validation, test or forecast.

    Returns
    -------
    instance : tuple[`tensors`] whith shape (inputs, known, date_inputs, date_labels), labels
        See :class:`autopycoin.dataset.generator.WindowGenerator` to get more informations
        abouyt each tensor.
    """

    if isinstance(dataset, tuple):
        if (
            len(dataset) != 2
            or len(dataset[0]) != 4
            or not isinstance(dataset[1], tf.Tensor)
        ):
            raise ValueError(
                "Accepts only tuple of shapes (inputs, labels) with inputs of lenght 4 and where labels is a tensor"
            )

        # test type
        dtypes = [output.dtype for output in dataset[0]] + [dataset[1].dtype]
        expected_types = [
            tf.float32,
            tf.float32,
            np.dtype("<U2"),
            np.dtype("<U2"),
            tf.float32,
        ]
        if dtypes != expected_types:
            raise ValueError(
                f"Accepts only tuple of types {expected_types}. got {dtypes}"
            )

        return dataset

    # dataset needs to be a tensorflow dataset or a tuple
    elif not isinstance(dataset, tf.data.Dataset):
        raise ValueError(
            f"Accepts only tensorflow dataset or tuple. got {type(dataset)}"
        )

    inputs, outputs = iter(dataset).get_next()

    (inputs, known, date_inputs, date_labels) = inputs
    labels = outputs

    # date_inputs and date_labels are bytes and need to be decoded to feed matplotlib plot
    date_inputs = date_inputs.numpy().astype("str")
    date_labels = date_labels.numpy().astype("str")

    return (inputs, known, date_inputs, date_labels), labels
