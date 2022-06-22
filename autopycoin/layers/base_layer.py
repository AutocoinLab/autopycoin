"""
Overloading Layers tensorflow object
"""

from typing import List, Union, Tuple
import tensorflow as tf

from autopycoin.constant import TENSOR_TYPE

from ..baseclass import AutopycoinBaseClass
from ..utils.data_utils import (
    convert_to_list,
    quantiles_handler
)


class QuantileLayer(tf.keras.layers.Layer, AutopycoinBaseClass):
    """Integrates a `quantiles` attribute to the layer.

    This layer aims to be inherited.
    During compilation if the model is a :class:`autopycoin.models.QuantileModel` it can propagate
    to this layer a `quantiles` attribute which can be added
    to its internal weights shape during building phase with the `get_additional_shapes` method.

    Usually, you will use this layer inside a :class:`autopycoin.models.QuantileModel` hence the transpose operation
    needed to fit the keras norm (see `post_processing` method) is not usefull here.
    Hence we created an `apply_quantiles_transpose` attribute accessible in constructor which decides
    if the layer has to transpose the outputs tensors and to convert them into :class:`autopycoin.extension_type.QuantileTensor`.
    by default, it is set to False but if you decides to use this layer inside `tf.keras.Model`
    set it to True.

    Attributes
    ----------
    has_quantiles : bool
        True if `quantiles` is not None else False. It is defined during compiling `method`.
        Default to False.
    quantiles : list[List[float]] or None
        It defines the quantiles used in the model.
        `quantiles` is a list of lists depending on the number of outputs the model computes.
        It is defined during compiling `method`.
        Default to None.
    n_quantiles : list[int] or int
        The number of quantiles the model computes.
        It is defined during compiling `method`.
        Default to 0.

    Notes
    -----
    .. code-block:: python

        def build(self, inputs_shape):
            self.get_additional_shapes(0) + output_shape # get the quantile shape and add it where you need it
    """

    def __init__(
        self, quantiles: Union[None, List[float]]=None, *args: list, **kwargs: dict
    ) -> None:

        super().__init__(*args, **kwargs)

        self._set_quantiles(quantiles)

    def _set_quantiles(
        self, quantiles: Union[None, list[Union[float, list[float]]]]
    ) -> None:
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self._has_quantiles = bool(quantiles)
        self._quantiles = quantiles_handler(quantiles) if quantiles is not None else quantiles
        self._n_quantiles = [len(self.quantiles)] if self.quantiles is not None else []

    def _propagate_quantiles(self):

        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], "_init_params"):
                self.layers[idx]._set_quantiles(
                    self.quantiles
                )  # pylint: disable=protected-access
                self.layers[idx]._propagate_quantiles()

    def _init_params(
        self,
        inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]],
        **kwargs: dict
    ) -> None:

        self._propagate_quantiles()

    @property
    def quantiles(self) -> List[List[float]]:
        """Return quantiles attribute."""

        return self._quantiles

    @property
    def n_quantiles(self) -> List[List[int]]:
        """Return the number of quantiles."""

        return self._n_quantiles

    @property
    def has_quantiles(self) -> bool:
        """Return True if quantiles exists else False."""

        return self._has_quantiles

    @property
    def additional_shape(self):
        return self.n_quantiles

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))


class UnivariateLayer(QuantileLayer):
    """Integrate a `n_variates` attribute to the layer.

    This layer aims to be inherited.
    During compilation if the model is a :class:`autopycoin.models.UnivariateModel` it can propagate
    to this layer a `n_variates` attribute which can be added
    to its internal weights shape during building phase with the `get_additional_shapes` method.
    This layer inherit from :class:`autopycoin.layers.QuantileLayer` then `get_additional_shapes` has also the `quantiles` dimension.

    Usually, you will use this layer inside a :class:`autopycoin.models.UnivariateModel` hence the transpose operation
    needed to fit the keras norm (see `post_processing` method) is not usefull here.
    Hence we created an `apply_multivariate_transpose` attribute accessible in constructor which decides
    if the layer has to transpose the outputs tensors and to convert them into :class:`autopycoin.extension_type.UnivariateTensor`.
    by default, it is set to False but if you decides to use this layer inside `tf.keras.Model`
    set it to True.

    Attributes
    ----------
    is_multivariate : bool
        True if the inputs rank is higher than 2. Default to False.
    n_variates : list[None | int]
        the number of variates in the inputs. Default to [].

    Notes
    -----
    .. code-block:: python

        def build(self, inputs_shape):
            self.get_additional_shapes(0) + output_shape # get the quantile shape and add it where you need it
    """

    def __init__(
        self, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self._is_multivariate = False
        self._n_variates = []

    def _init_params(
        self,
        inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]],
        n_variates: Union[None, int, List[int]] = None,
        is_multivariate: Union[None, bool] = None,
    ) -> None:
        """Initialize attributes related to univariate model.

        It is called before `build`.
        Three steps are done:
        - Filter the first shape in case of multiple inputs tensors.
        - Initialize attributes: `is_multivariate`, `n_variates`.
        - Add the n_variates dimension to `additional_shape`.
        """

        if isinstance(inputs_shape, list):
            inputs_shape = inputs_shape[0]

        super()._init_params(
            inputs_shape=inputs_shape
        )

        self._set_multivariates(inputs_shape=inputs_shape, n_variates=n_variates, is_multivariate=is_multivariate)
        self._propagate_multivariates(inputs_shape)

    def _set_multivariates(self,
        inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]],
        n_variates: Union[None, int, List[int]] = None,
        is_multivariate: Union[None, bool] = None,
        ):

        self._is_multivariate = is_multivariate or bool(inputs_shape.rank > 2)
        if self.is_multivariate:
            self._n_variates = convert_to_list(n_variates or inputs_shape[-1])

    def _propagate_multivariates(self, inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], "_init_params"):
                self.layers[idx]._set_multivariates(
                    inputs_shape,
                    self.n_variates,
                    self.is_multivariate
                )
                self.layers[idx]._propagate_multivariates(
                    inputs_shape
                )  # pylint: disable=protected-access

    @property
    def is_multivariate(self) -> bool:
        return self._is_multivariate

    @property
    def n_variates(self) -> List[Union[None, int]]:
        return self._n_variates

    @property
    def additional_shape(self):
        return self.n_quantiles + self.n_variates
