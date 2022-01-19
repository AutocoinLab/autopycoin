"""Checks used to modified inputs."""

from typing import Any, Union, Tuple, List
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx


def avoid_infinity(tensor: tf.Tensor) -> tf.Tensor:
    """
    This function is used to verify infinite values and to mask them.

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


def example_handler(dataset: tf.data.Dataset, window_generator: Any) -> Tuple[tf.Tensor]:
    """
    Convenient function which extract one instance of a
    `WindowGenerator` train, validation, test or a forecast dataset.

    Parameters
    ----------
    dataset : `WindowGenerator datasets` or tuple[`tensor`]
        `WindowGenerator` datasets train, validation, test or forecast.
    window_generator: :class:`autopycoin.dataset.WindowGenerator`
        A :class:`autopycoin.dataset.WindowGenerator` instance.

    Returns
    -------
    instance : tuple[`tensors`] whith shape (inputs, known, date_inputs, date_labels), labels
        See :class:`autopycoin.dataset.generator.WindowGenerator` to get more informations
        abouyt each tensor.
    """

    # If tuple we make sure it is composed by tensors
    if isinstance(dataset, tuple):
        index = [
            True,
            bool(window_generator.known_columns),
            bool(window_generator.date_columns),
            bool(window_generator.date_columns),
        ]
        if (
            len(dataset) != 2
            or sum(index) != sum(True for tensor in dataset[0] if tensor is not None)
            or not isinstance(dataset[1], tf.Tensor)
        ):
            raise ValueError(
                f"""Accepts only a tuple of shapes (inputs, labels) with the inputs composed by {sum(index)} tensors.
                Got an inputs of {len(dataset[0])} tensors and a labels component of type {type(dataset[1])}."""
            )

        inputs, labels = dataset
        # Fill by None value to build an inputs tuple of shape (inputs, known, date_inputs, date_labels)
        inputs = fill_none(inputs, index=index)
        dtypes = [inp.dtype for inp in inputs if inp is not None] + [dataset[1].dtype]

        expected_types = [tf.float32, np.dtype("<U2")]

        for dtype in dtypes:
            if dtype != np.dtype("<U2") and dtype != tf.float32:
                raise ValueError(
                    f"Accepts only tensors of types {expected_types}. got {dtypes}"
                )
        return inputs, labels

    # dataset needs to be a tensorflow dataset or a tuple
    elif not isinstance(dataset, tf.data.Dataset):
        raise ValueError(
            f"Accepts only tensorflow dataset or tuple. got {type(dataset)}"
        )

    inputs, labels = iter(dataset).get_next()

    # Fill by None value to build an inputs tuple of shape (inputs, known, date_inputs, date_labels)
    inputs = fill_none(
        inputs,
        index=[
            True,
            bool(window_generator.known_columns),
            bool(window_generator.date_columns),
            bool(window_generator.date_columns),
        ],
    )

    (inputs, known, date_inputs, date_labels) = inputs

    # date_inputs and date_labels are bytes and need to be decoded to feed matplotlib plot
    if not isinstance(date_inputs, type(None)):
        date_inputs = date_inputs.numpy().astype("str")
    if not isinstance(date_labels, type(None)):
        date_labels = date_labels.numpy().astype("str")

    return (inputs, known, date_inputs, date_labels), labels


def fill_none(
    inputs: Union[tuple, tf.Tensor], max_value: int = 4, index: List[bool] = None
) -> Tuple[tf.Tensor]:
    """Fill the inputs tuple by None values."""
    # if tensor then it becomes a list of lists
    inputs = list(inputs) if isinstance(inputs, tuple) else [inputs]
    if not index:
        return tuple(
            None if value > len(inputs) - 1 else inputs[value]
            for value in range(max_value)
        )
    return tuple(
            None if not index[value] else inputs.pop(0) for value in range(max_value)
        )


def convert_to_list(to_convert: Any) -> list:
    """Wrap the object with a list.
    If a list is provided, it doesn't wrap it."""
    return [to_convert] if not isinstance(to_convert, list) else to_convert


def transpose_first_to_last(inputs: tf.Tensor):
    """transpose the first dimension to the last position."""
    #TODO: unit testing
    perm = tf.concat([tf.range(1, tf.rank(inputs)), [0]], axis=0)
    return tf.transpose(inputs, perm=perm)


def transpose_last_to_first(inputs: tf.Tensor):
    """transpose the last dimension to the first position."""
    #TODO: unit testing
    perm = tf.concat([[tf.rank(inputs) - 1], tf.range(tf.rank(inputs) - 1)], axis=0)
    return tf.transpose(inputs, perm=perm)

def features(inputs, features_slice, columns_index):
    """Return an input and output date tensors from the features tensor."""
    feature_length = features_slice.stop - features_slice.start
    feature = tf.stack([inputs[..., features_slice, index] for index in columns_index], axis=-1)
    feature.set_shape([None, feature_length, len(columns_index)])
    return feature

def date_features(inputs, features_slice, columns_index) -> tf.Tensor:
    """Return an input and output date tensors from the features tensor."""
    date = features(inputs, features_slice, columns_index)
    date = tf.cast(date, tf.int32)
    date = tf.strings.as_string(date)
    return tf.strings.reduce_join(date, separator="-", axis=-1, keepdims=True)