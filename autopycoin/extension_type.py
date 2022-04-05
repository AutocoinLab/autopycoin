"Defines tensor types"

from typing import Union, List

import tensorflow as tf
from tensorflow.python.ops import array_ops
import keras

from . import losses


class QuantileTensor(tf.experimental.ExtensionType):
    """Extension type to introduce quantiles in tensor"""

    values: tf.Tensor
    quantiles: bool
    shape: tf.TensorShape
    dtype: tf.DType

    def __init__(self, values, quantiles: bool, shape=None, dtype=None):
        self.values = values
        self.quantiles = quantiles
        self.shape = shape or self.values.shape
        self.dtype = dtype or self.values.dtype

    def __getitem__(self, key):
        return QuantileTensor(self.values.__getitem__(key), quantiles=self.quantiles)

    def __sub__(self, tensor):
        return QuantileTensor(self.values.__sub__(tensor), quantiles=self.quantiles)

    @property
    def rank(self):
        return self.values.rank


class UnivariateTensor(QuantileTensor):
    """Extension type to introduce multivariates in tensor"""

    values: tf.Tensor
    quantiles: bool
    multivariates: bool
    shape: tf.TensorShape
    dtype: tf.DType

    def __init__(
        self,
        values: Union[QuantileTensor, tf.Tensor],
        multivariates: bool,
        quantiles: bool = None,
        shape=None,
        dtype=None,
    ):
        is_quantile_tensor = isinstance(values, QuantileTensor)
        self.values = values.values if is_quantile_tensor else values
        self.quantiles = values.quantiles if is_quantile_tensor else quantiles
        self.multivariates = multivariates
        self.shape = shape or self.values.shape
        self.dtype = dtype or self.values.dtype

    def __getitem__(self, key):
        return UnivariateTensor(
            self.values.__getitem__(key),
            quantiles=self.quantiles,
            multivariates=self.multivariates,
        )

    def __sub__(self, tensor):
        return UnivariateTensor(
            self.values.__sub__(tensor),
            quantiles=self.quantiles,
            multivariates=self.multivariates,
        )


# dispatching tensorflow API for UnivariateTensor and QuantileTensor
@tf.experimental.dispatch_for_api(tf.linalg.matmul)
def matmul(
    a: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    b: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None,
):
    x_values = a.values if isinstance(a, (UnivariateTensor, QuantileTensor)) else a
    y_values = b.values if isinstance(b, (UnivariateTensor, QuantileTensor)) else b
    x_is_univariate = isinstance(a, UnivariateTensor)
    x_is_quantile = isinstance(a, QuantileTensor)
    if x_is_univariate or isinstance(
        b, UnivariateTensor
    ):
        a = a if x_is_univariate else b
        return UnivariateTensor(
            tf.matmul(
                x_values,
                y_values,
                transpose_a,
                transpose_b,
                adjoint_a,
                adjoint_b,
                a_is_sparse,
                b_is_sparse,
                output_type,
                name,
            ),
            quantiles=a.quantiles,
            multivariates=a.multivariates,
            shape=a.shape,
        )
    elif x_is_quantile or isinstance(
        b, QuantileTensor
    ):
        a = a if x_is_quantile else b
        return QuantileTensor(
            tf.matmul(
                x_values,
                y_values,
                transpose_a,
                transpose_b,
                adjoint_a,
                adjoint_b,
                a_is_sparse,
                b_is_sparse,
                output_type,
                name,
            ),
            quantiles=a.quantiles,
            shape=a.shape,
        )
    return tf.matmul(
        x_values,
        y_values,
        transpose_a,
        transpose_b,
        adjoint_a,
        adjoint_b,
        a_is_sparse,
        b_is_sparse,
        output_type,
        name,
    )


@tf.experimental.dispatch_for_api(tf.convert_to_tensor)
def convert_to_tensor(
    value: Union[QuantileTensor, UnivariateTensor],
    dtype=None,
    dtype_hint=None,
    name=None,
):
    if isinstance(value, UnivariateTensor):
        return UnivariateTensor(
            tf.convert_to_tensor(
                value.values, dtype=dtype, dtype_hint=dtype_hint, name=name
            ),
            quantiles=value.quantiles,
            multivariates=value.multivariates,
        )
    elif isinstance(value, QuantileTensor):
        return QuantileTensor(
            tf.convert_to_tensor(
                value.values, dtype=dtype, dtype_hint=dtype_hint, name=name
            ),
            quantiles=value.quantiles,
        )


@tf.experimental.dispatch_for_api(tf.squeeze)
def convert_to_tensor(
    input: Union[QuantileTensor, UnivariateTensor], axis=None, name=None
):
    if isinstance(input, UnivariateTensor):
        return UnivariateTensor(
            tf.squeeze(input.values, axis=axis, name=name),
            quantiles=input.quantiles,
            multivariates=input.multivariates,
        )
    elif isinstance(input, QuantileTensor):
        return QuantileTensor(
            tf.squeeze(input.values, axis=axis, name=name), quantiles=input.quantiles
        )


@tf.experimental.dispatch_for_api(tf.concat)
def concat(
    values: List[Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable]],
    axis,
    name="concat",
):
    val = [
        v.values if isinstance(v, (QuantileTensor, UnivariateTensor)) else v
        for v in values
    ]
    quantiles = any(
        v.quantiles if isinstance(v, (QuantileTensor, UnivariateTensor)) else False
        for v in values
    )
    if any(isinstance(v, UnivariateTensor) for v in values):
        multivariates = any(
            v.multivariates
            if isinstance(v, (QuantileTensor, UnivariateTensor))
            else False
            for v in values
        )
        return UnivariateTensor(
            tf.concat(val, axis=axis, name=name),
            quantiles=quantiles,
            multivariates=multivariates,
        )
    elif any(isinstance(v, QuantileTensor) for v in values):
        return QuantileTensor(tf.concat(val, axis=axis, name=name), quantiles=quantiles)

    return tf.concat(values, axis, name)


@tf.experimental.dispatch_for_api(tf.rank)
def rank(input: Union[QuantileTensor, UnivariateTensor], name=None):
    return tf.rank(input.values, name=name)


@tf.experimental.dispatch_for_api(array_ops.size)
def size_v1(
    input: Union[QuantileTensor, UnivariateTensor], name=None, out_type=tf.int32
):
    return tf.size(input.values, out_type=out_type, name=name)


@tf.experimental.dispatch_for_api(tf.size)
def size(input: Union[QuantileTensor, UnivariateTensor], out_type=tf.int32, name=None):
    return tf.size(input.values, out_type=out_type, name=name)


@tf.experimental.dispatch_for_api(tf.shape)
def shape(input: Union[QuantileTensor, UnivariateTensor], out_type=tf.int32, name=None):
    return tf.shape(input.values, out_type=out_type, name=name)


@tf.experimental.dispatch_for_api(tf.argmax)
def argmax(
    input: Union[QuantileTensor, UnivariateTensor],
    axis=None,
    output_type=tf.int64,
    name=None,
):
    return tf.argmax(input.values, axis=axis, output_type=output_type, name=name)


@tf.experimental.dispatch_for_api(tf.add_n)
def add_n(
    inputs: List[Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable]],
    name=None,
):
    val = [
        v.values if isinstance(v, (QuantileTensor, UnivariateTensor)) else v
        for v in inputs
    ]
    quantiles = any(
        [
            v.quantiles if isinstance(v, (QuantileTensor, UnivariateTensor)) else False
            for v in inputs
        ]
    )
    if any([isinstance(v, UnivariateTensor) for v in inputs]):
        multivariates = any(
            [
                v.multivariates
                if isinstance(v, (QuantileTensor, UnivariateTensor))
                else False
                for v in inputs
            ]
        )
        return UnivariateTensor(
            tf.add_n(val, name=name), quantiles=quantiles, multivariates=multivariates
        )
    elif any([isinstance(v, QuantileTensor) for v in inputs]):
        return QuantileTensor(tf.add_n(val, name=name), quantiles=quantiles)
    return tf.add_n(inputs, name)


@tf.experimental.dispatch_for_api(tf.round)
def round(x: Union[QuantileTensor, UnivariateTensor], name=None):
    if isinstance(x, UnivariateTensor):
        return UnivariateTensor(
            tf.round(x.values, name=name),
            quantiles=x.quantiles,
            multivariates=x.multivariates,
        )
    elif isinstance(x, QuantileTensor):
        return QuantileTensor(tf.round(x.values, name=name), quantiles=x.quantiles)


@tf.experimental.dispatch_for_api(tf.transpose)
def transpose(
    a: Union[QuantileTensor, UnivariateTensor],
    perm=None,
    conjugate=False,
    name="transpose",
):
    if isinstance(a, UnivariateTensor):
        return UnivariateTensor(
            tf.transpose(a.values, perm=perm, conjugate=conjugate, name=name),
            quantiles=a.quantiles,
            multivariates=a.multivariates,
        )
    elif isinstance(a, QuantileTensor):
        return QuantileTensor(
            tf.transpose(a.values, perm=perm, conjugate=conjugate, name=name),
            quantiles=a.quantiles,
        )


@tf.experimental.dispatch_for_api(tf.math.reduce_mean)
def reduce_mean(
    input_tensor: Union[
        List[Union[QuantileTensor, UnivariateTensor]],
        Union[QuantileTensor, UnivariateTensor],
    ],
    axis=None,
    keepdims=False,
    name=None,
):
    if isinstance(input_tensor, list):
        input_tensor = [input.values for input in input_tensor]
        return tf.math.reduce_mean(
            input_tensor, axis=axis, keepdims=keepdims, name=name
        )
    return tf.math.reduce_mean(
        input_tensor.values, axis=axis, keepdims=keepdims, name=name
    )


@tf.experimental.dispatch_for_api(tf.math.reduce_sum)
def reduce_sum(
    input_tensor: Union[QuantileTensor, UnivariateTensor],
    axis=None,
    keepdims=False,
    name=None,
):
    return tf.math.reduce_sum(
        input_tensor.values, axis=axis, keepdims=keepdims, name=name
    )


@tf.experimental.dispatch_for_api(tf.identity)
def identity(
    input: List[Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable]],
    name=None,
):
    val = [
        v.values if isinstance(v, (QuantileTensor, UnivariateTensor)) else v
        for v in input
    ]
    quantiles = any(
        [
            v.quantiles if isinstance(v, (QuantileTensor, UnivariateTensor)) else False
            for v in input
        ]
    )
    if any([isinstance(v, UnivariateTensor) for v in input]):
        multivariates = any(
            [
                v.multivariates
                if isinstance(v, (QuantileTensor, UnivariateTensor))
                else False
                for v in input
            ]
        )
        return UnivariateTensor(
            tf.identity(val, name=name),
            quantiles=quantiles,
            multivariates=multivariates,
        )
    elif any([isinstance(v, QuantileTensor) for v in input]):
        return QuantileTensor(tf.identity(val, name=name), quantiles=quantiles)
    return tf.identity(input, name)


@tf.experimental.dispatch_for_api(keras.backend.maximum)
def maximum(
    x: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    y: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
):
    x = x.values if isinstance(x, (QuantileTensor, UnivariateTensor)) else x
    y = y.values if isinstance(y, (QuantileTensor, UnivariateTensor)) else y
    return tf.maximum(x, y)


@tf.experimental.dispatch_for_api(tf.keras.losses.mean_absolute_percentage_error)
def mean_absolute_percentage_error(
    y_true: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    y_pred: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
):
    y_true = (
        y_true.values
        if isinstance(y_true, (QuantileTensor, UnivariateTensor))
        else y_true
    )
    y_pred = (
        y_pred.values
        if isinstance(y_pred, (QuantileTensor, UnivariateTensor))
        else y_pred
    )
    return tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)


@tf.experimental.dispatch_for_api(losses.smape)
def symetric_mean_absolute_percentage_error(
    y_true: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    y_pred: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
):
    y_true = (
        y_true.values
        if isinstance(y_true, (QuantileTensor, UnivariateTensor))
        else y_true
    )
    y_pred = (
        y_pred.values
        if isinstance(y_pred, (QuantileTensor, UnivariateTensor))
        else y_pred
    )
    return losses.smape(y_true, y_pred)


def dispatch(tensor, fn, **kwargs):
    if isinstance(tensor, UnivariateTensor):
        return UnivariateTensor(
            fn(tensor.values, **kwargs),
            quantiles=tensor.quantiles,
            multivariates=tensor.multivariates,
        )
    elif isinstance(tensor, QuantileTensor):
        return QuantileTensor(fn(tensor.values, **kwargs), quantiles=tensor.quantiles)
    return fn(tensor, **kwargs)


@tf.experimental.dispatch_for_api(tf.nn.relu)
def relu(features: Union[UnivariateTensor, QuantileTensor], name=None):
    dispatch(features, tf.nn.relu, name=name)


@tf.experimental.dispatch_for_api(tf.nn.dropout)
def dropout(
    x: Union[UnivariateTensor, QuantileTensor],
    rate,
    noise_shape=None,
    seed=None,
    name=None,
):
    dispatch(x, tf.nn.dropout, rate=rate, noise_shape=noise_shape, seed=seed, name=name)


@tf.experimental.dispatch_for_unary_elementwise_apis(
    Union[QuantileTensor, UnivariateTensor]
)
def tensor_unary_elementwise_api_handler(api_func, x):
    if isinstance(x, UnivariateTensor):
        return UnivariateTensor(
            api_func(x.values), quantiles=x.quantiles, multivariates=x.multivariates
        )
    elif isinstance(x, QuantileTensor):
        return QuantileTensor(api_func(x.values), quantiles=x.quantiles)


@tf.experimental.dispatch_for_binary_elementwise_apis(
    Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
    Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable],
)
def tensor_binary_elementwise_api_handler(api_func, x, y):
    x_values = x.values if isinstance(x, (UnivariateTensor, QuantileTensor)) else x
    y_values = y.values if isinstance(y, (UnivariateTensor, QuantileTensor)) else y
    x_is_univariate = isinstance(x, UnivariateTensor)
    x_is_quantile = isinstance(x, QuantileTensor)

    if x_is_univariate or isinstance(
        y, UnivariateTensor
    ):
        x = x if x_is_univariate else y
        return UnivariateTensor(
            api_func(x_values, y_values),
            quantiles=x.quantiles,
            multivariates=x.multivariates,
            shape=x.shape,
        )
    elif x_is_quantile or isinstance(
        y, QuantileTensor
    ):
        x = x if x_is_quantile else y
        return QuantileTensor(
            api_func(x_values, y_values), quantiles=x.quantiles, shape=x.shape
        )
    return api_func(x_values, y_values)
