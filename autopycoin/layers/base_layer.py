"""
Overloading Layers tensorflow object
"""

import tensorflow as tf


class BaseLayer(tf.keras.layers.Layer, metaclass=AutopycoinMetaModel):
    """Base model which defines pre/post-processing methods to override.

    Currently, four wrappers can be overriden:
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data
    - metrics_wrapper : Preprocess y_true or y_pred
    - losses_wrapper : Preprocess y_true or y_pred

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


class QuantileLayer(tf.keras.layers.Layer):
    """
    Override tensorflow Layer class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hence it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.

    See tensorflow documentation for more details.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]

    def _set_quantiles(self, value, additional_shapes=None, n_quantiles=None):
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self._built = False
        self._quantiles = value
        self._additional_shapes = additional_shapes
        self._n_quantiles = n_quantiles

    def get_additional_shapes(self, index):
        """Return the shape to add to your layers."""

        try:
            return self._additional_shapes[index]
        except IndexError:
            return []

    @property
    def quantiles(self):
        """Return the `quantiles` attribute."""

        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""

        return self._n_quantiles


class UnivariateLayer(QuantileLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_multivariate_transpose = True
        self._init_multivariates_params = False

        self._n_variates = []
        self._is_multivariate = False

    def init_params(self, input_shape, n_variates, is_multivariate, additional_shapes):

        self._n_variates = n_variates
        self._additional_shapes = additional_shapes
        self._is_multivariate = is_multivariate
        self._init_multivariates_params = True

    def get_additional_shapes(self, index):
        if not self._init_multivariates_params:
            raise AssertionError('Please call `init_params` at the beginning of build.')
        return super().get_additional_shapes(index)

    @property
    def is_multivariate(self):
        return self._is_multivariate

    @property
    def n_variates(self):
        return self._n_variates
