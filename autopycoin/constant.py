""" Defines constant """

from typing import Union, List
import tensorflow as tf
from .extension_type import QuantileTensor, UnivariateTensor

TENSOR_TYPE = Union[
    List[Union[tf.Tensor, QuantileTensor, UnivariateTensor]],
    tf.Tensor,
    QuantileTensor,
    UnivariateTensor,
]
