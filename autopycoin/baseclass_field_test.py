# pylint: skip-file

"""Tests for baseclass_field."""

import typing
from absl.testing import parameterized
import pandas as pd

import tensorflow as tf
from tensorflow.python.util import type_annotations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor

from . import baseclass_field


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeFieldTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            # Without default values:
            ("x", int),
            ("f", float),
            ("t", ops.Tensor),
            ("d", pd.DataFrame),
            # With default values:
            ("x", int, 33),
            ("y", float, 33.8),
            ("t", ops.Tensor, lambda: constant_op.constant([[1, 2], [3, 4]])),
            (
                "r",
                ragged_tensor.RaggedTensor,
                lambda: ragged_factory_ops.constant([[1, 2], [3]]),
            ),
            ("union", typing.Union[int, float], 4),
            ("seq", typing.Tuple[typing.Union[int, float], ...], (33, 12.8, 9, 0)),
            (
                "seq",
                typing.Tuple[typing.Union[int, float], ...],
                [33, 12.8, 9, 0],
                (33, 12.8, 9, 0),
            ),
        ]
    )
    def testConstruction(
        self,
        name,
        value_type,
        default=baseclass_field.AutopycoinField.NO_DEFAULT,
        converted_default=None,
    ):
        if callable(default):
            default = default()  # deferred construction (contains tensor)
        field = baseclass_field.AutopycoinField(name, value_type, default)
        if converted_default is not None:
            default = converted_default
        self.assertEqual(field.name, name)
        self.assertEqual(field.value_type, value_type)
        if isinstance(default, (ops.Tensor, ragged_tensor.RaggedTensor)):
            self.assertAllEqual(field.default, default)
        else:
            self.assertEqual(field.default, default)

    @parameterized.parameters(
        [
            ("i", int, 8.3, "default value for i: expected int, got 8.3"),
            (
                "x",
                int,
                "hello world",
                "default value for x: expected int, got 'hello world'",
            ),
            (
                "seq",
                typing.Tuple[typing.Union[int, float], ...],
                [33, 12.8, "zero"],
                "expected float, got 'zero'",
            ),
        ]
    )
    def testConstructionError(self, name, value_type, default, error):
        if callable(default):
            default = default()  # deferred construction (contains tensor)
        with self.assertRaisesRegex(TypeError, error):
            baseclass_field.AutopycoinField(name, value_type, default)

    @parameterized.parameters(
        [
            (
                "AutopycoinField(name='i', value_type=<class 'int'>, "
                "default=AutopycoinField.NO_DEFAULT)",
                "i",
                int,
            ),
            (
                "AutopycoinField(name='x', value_type=typing.Tuple"
                "[typing.Union[str, int], ...], default=AutopycoinField.NO_DEFAULT)",
                "x",
                typing.Tuple[typing.Union[str, int], ...],
            ),
            (
                "AutopycoinField(name='j', value_type=<class 'int'>, default=3)",
                "j",
                int,
                3,
            ),
        ]
    )
    def testRepr(
        self,
        expected,
        name,
        value_type,
        default=baseclass_field.AutopycoinField.NO_DEFAULT,
    ):
        field = baseclass_field.AutopycoinField(name, value_type, default)
        self.assertEqual(repr(field), expected)


class TypeAnnotationsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            (typing.Union[int, float], "Union"),
            (typing.Tuple[int, ...], "Tuple"),
            (typing.Tuple[int, float, float], "Tuple"),
            (typing.Mapping[int, float], "Mapping"),
            (typing.Union[typing.Tuple[int], typing.Tuple[int, ...]], "Union"),
            # These predicates return False for Generic types w/ no parameters:
            (typing.Union, None),
            (typing.Tuple, None),
            (typing.Mapping, None),
            (int, None),
            (12, None),
        ]
    )
    def testGenericTypePredicates(self, tp, expected):
        self.assertEqual(baseclass_field.is_generic_union(tp), expected == "Union")
        self.assertEqual(baseclass_field.is_generic_tuple(tp), expected == "Tuple")
        self.assertEqual(baseclass_field.is_generic_mapping(tp), expected == "Mapping")

    @parameterized.parameters(
        [
            (typing.Union[int, float], (int, float)),
            (typing.Tuple[int, ...], (int, Ellipsis)),
            (typing.Tuple[int, float, float], (int, float, float,),),
            (typing.Mapping[int, float], (int, float)),
            (
                typing.Union[typing.Tuple[int], typing.Tuple[int, ...]],
                (typing.Tuple[int], typing.Tuple[int, ...]),
            ),
        ]
    )
    def testGetGenericTypeArgs(self, tp, expected):
        self.assertEqual(type_annotations.get_generic_type_args(tp), expected)

    def testIsForwardRef(self):
        tp = typing.Union["B", int]
        tp_args = type_annotations.get_generic_type_args(tp)
        self.assertTrue(type_annotations.is_forward_ref(tp_args[0]))
        self.assertFalse(type_annotations.is_forward_ref(tp_args[1]))
