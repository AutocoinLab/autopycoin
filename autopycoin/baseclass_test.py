# pylint: skip-file

"""Tests for autopycoin base class"""


from absl.testing import parameterized
from autopycoin.baseclass import AutopycoinBaseClass

import tensorflow as tf
from tensorflow.python.framework import test_util

from . import AutopycoinBaseClass


class ExampleClass(AutopycoinBaseClass):
    """Example subclass of AutopycoinBaseClass, used for testing."""

    def __init__(self, input1: int, input2: tf.Tensor, input3: int = 4):
        self._input1 = input1
        self._input2 = input2
        self.input3 = input3

    @property
    def input1(self):
        return self._input1

    @property
    def input2(self):
        return self._input2

    @classmethod
    def from_full_tensor(cls, values: int):
        return values

    # A dummy example to test support of staticmethod
    @staticmethod
    def doc_link():
        return "http://example.com/ExampleClass"

    def _val___init__(self, output, method_name, *args, **kwargs):
        assert self.input1 > 2
        self._input2.shape.assert_is_compatible_with(self._input2.shape)


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeTest(tf.test.TestCase, parameterized.TestCase):
    def testValueWhenDefaultProvided(self):
        with self.assertRaisesRegexp(
            TypeError, "value for input3 in __init__: expected int, got None"
        ):
            ExampleClass(3, tf.constant([1]), None)
            ExampleClass(3, tf.constant([1]), input3=None)

    def testAttributeAccessors(self):
        mt1 = ExampleClass(3, tf.constant([1]))

        self.assertIsInstance(mt1.input1, int)
        self.assertAllEqual(mt1.input1, 3)
        self.assertIsInstance(mt1.input2, tf.Tensor)
        self.assertAllEqual(mt1.input2, tf.constant([1]))

    def testNoAnnotation(self):
        with self.assertRaises(ValueError):

            class ExampleClass2(AutopycoinBaseClass):
                """Example subclass of AutopycoinBaseClass, used for testing."""

                def __init__(self, input1, input2: tf.Tensor):
                    self._input1 = input1
                    self._input2 = input2

    def testTypeValues(self):
        with self.assertRaises(TypeError):
            ExampleClass(3.0, tf.constant([1]))

    def testValidate(self):
        with self.assertRaises(AssertionError):
            ExampleClass(1, tf.constant([1]))
