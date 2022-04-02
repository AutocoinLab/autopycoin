"""
Overloading Layers tensorflow object
"""

import abc
import tensorflow as tf

from ..extension_type import QuantileTensor, UnivariateTensor
from ..utils.data_utils import convert_to_list, transpose_first_to_last, transpose_first_to_second_last, transpose_last_to_first


class AutopycoinMetaLayer(abc.ABCMeta):
    """Metaclass for autopycoin models."""

    def __init__(cls, name, bases, namespace):
        cls.build = _wrap_build(cls.build)
        cls.call = _wrap_call(cls.call)

        super().__init__(name, bases, namespace)


def _wrap_build(fn):
    """Wrap the build method with a init_params function"""

    def build_wrapper(self, inputs_shape):
        self.init_params(inputs_shape)
        return fn(self, inputs_shape)
    return build_wrapper


def _wrap_call(fn):
    """Wrap the call method with a _preprocessing and _post_processing methods"""

    def call_wrapper(self, inputs, *args, **kwargs):
        inputs = self._preprocessing_wrapper(inputs)
        outputs = fn(self, inputs, *args, **kwargs)
        outputs = self._post_processing_wrapper(outputs)
        return outputs
    return call_wrapper

class BaseLayer(tf.keras.layers.Layer, metaclass=AutopycoinMetaLayer):
    """Base model which defines pre/post-processing methods to override.

    Currently, two wrappers can be overriden:
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data

    `Compute_output_shape` needs also to be implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preprocessing_wrapper(self, inputs):
        return self.preprocessing(inputs)

    def preprocessing(self, inputs):
        """Public API to apply preprocessing logics to your inputs data."""

        raise NotImplementedError('`preprocessing` has to be overriden.')

    def _post_processing_wrapper(self, outputs):
        """Post-processing wrapper."""

        outputs_is_nested = tf.nest.is_nested(outputs)
        outputs = tuple(outputs) if outputs_is_nested else (outputs,)
        outputs = tf.nest.map_structure(lambda output: self.post_processing(output), outputs)
    
        return outputs[0] if len(outputs) == 1 else outputs

    def post_processing(self, output):
        """Public API to apply post-processing logics to your outputs data."""

        raise NotImplementedError('`post_processing` has to be overriden.')


class QuantileLayer(BaseLayer):
    """
    Override tensorflow Layer class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hence it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.

    See tensorflow documentation for more details.
    """

    def __init__(self, apply_quantiles_transpose: bool=False, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.apply_quantiles_transpose = apply_quantiles_transpose
        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]

    def _set_quantiles(self, value, additional_shapes=None, n_quantiles=None):
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self._built = False
        self._has_quantiles = True
        self._quantiles = value
        self._additional_shapes = additional_shapes or [[len(q)] for q in self.quantiles]
        self._n_quantiles = n_quantiles or self._additional_shapes.copy()

    def preprocessing(self, inputs):
        """No preprocessing for `QuantileModel`"""
        return inputs

    def post_processing(self, outputs):
        """Convert the outputs to `QuantileTensor` and apply transpose operation.

        The quantiles dimension is put to the last dimension to fit with keras norms.
        There is a difference with the Model implementation, we can't check with losses
        if they have a quantile attribute hence `apply_quantiles_transpose` is set to False by default
        and if you need to implement a layer with transpose operation you have to set it to change that.
        """

        if self.apply_quantiles_transpose:
            outputs = transpose_first_to_last(outputs)
            if outputs.shape[-1] == 1:
                outputs = tf.squeeze(outputs, axis=-1)
            return QuantileTensor(outputs, quantiles=True)
        return QuantileTensor(outputs, quantiles=False)

    def init_params(self, input_shape, **kwargs):
        pass

    def get_additional_shapes(self, index):
        """Return the shape to add to your layers."""

        try:
            return self._additional_shapes[index]
        except IndexError:
            return []

    def init_params(self, input_shape, **kwargs):
        pass

    @property
    def quantiles(self):
        """Return quantiles attribute."""

        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""

        return self._n_quantiles

    @property
    def has_quantiles(self):
        """Return True if quantiles exists else False."""

        return self._has_quantiles


class UnivariateLayer(QuantileLayer):
    def __init__(self, apply_multivariate_transpose:bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.apply_multivariate_transpose = apply_multivariate_transpose
        self._init_multivariates_params = False

        self._n_variates = []
        self._is_multivariate = False

    def preprocessing(self, inputs):
        """Init the multivariates attributes and transpose the `nvariates` dimension in first position."""

        if self.apply_multivariate_transpose and self.is_multivariate:
            return tf.nest.map_structure(transpose_last_to_first, inputs)
        return inputs

    def post_processing(self, outputs):
        outputs = super().post_processing(outputs)
        if self.apply_multivariate_transpose:
            if self.is_multivariate:
                outputs = tf.nest.map_structure(lambda outputs: transpose_first_to_second_last(outputs) if outputs.quantiles else transpose_first_to_last(outputs), outputs)
                outputs = tf.nest.map_structure(convert_to_univariate_tensor(multivariates=True), outputs)
                return outputs
            return tf.nest.map_structure(convert_to_univariate_tensor(multivariates=False), outputs)
        return tf.nest.map_structure(convert_to_univariate_tensor(multivariates=False), outputs)

    def init_params(self, input_shape, n_variates=None, is_multivariate=None, additional_shapes=None):
        """Initialize attributes related to univariate model.
        
        It is called before `build`.
        Three steps are done:
        - Filter the first shape in case of multiple inputs tensors.
        - Initialize attributes: `is_multivariate`, `n_variates`.
        - Add the n_variates dimension to `additional_shape` and propagate these attributes
        to the internal layers.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor.
        """

        if not self._init_multivariates_params:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0]

            self._init_multivariates_params = True
            self.set_is_multivariate(input_shape, is_multivariate)
            self.set_n_variates(input_shape, n_variates)
            self.extend_additional_shape(additional_shapes)

    def set_is_multivariate(self, input_shape, is_multivariate=None):
        """Initiate `is_multivariate` attribute"""

        self._is_multivariate = is_multivariate or bool(input_shape.rank > 2)

    def set_n_variates(self, input_shape, n_variates=None):
        """Initiate `n_variates` attribute"""

        if self.is_multivariate:
            self._n_variates = convert_to_list(n_variates or input_shape[-1])

    def get_additional_shapes(self, index):
        if not self._init_multivariates_params:
            raise AssertionError('Please call `init_univariate_params` at the beginning of build.')
        return super().get_additional_shapes(index)

    def extend_additional_shape(self, additional_shapes=None):
        self._additional_shapes = additional_shapes or [s + self.n_variates for s in self._additional_shapes]

    @property
    def is_multivariate(self):
        return self._is_multivariate

    @property
    def n_variates(self):
        return self._n_variates


def convert_to_univariate_tensor(multivariates):
    def fn(tensor):
        return UnivariateTensor(values=tensor, multivariates=multivariates)
    return fn
