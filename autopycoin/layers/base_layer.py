"""
Overloading Layers tensorflow object
"""

from typing import List, Union
import tensorflow as tf

from ..extension_type import QuantileTensor, UnivariateTensor
from ..utils.data_utils import (
    convert_to_list,
    transpose_first_to_last,
    transpose_first_to_second_last,
    transpose_last_to_first,
)
from .. import AutopycoinBaseLayer
from ..constant import TENSOR_TYPE


# TODO: Unit test
class BaseLayer(tf.keras.layers.Layer, AutopycoinBaseLayer):
    """Base layer which defines pre/post-processing methods to override.

    This layer aims to be inherited and brings four functionality.
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data
    - init_params : initialize parameters before `build` method
    This three wrappers have to be overriden
    - Typing check.
    """

    NOT_INSPECT = ["build", "call"]

    def __init__(self, *args: list, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)

    def _preprocessing_wrapper(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        return self.preprocessing(inputs)

    def preprocessing(self, inputs: TENSOR_TYPE) -> None:
        """Public API to apply preprocessing logics to your inputs data."""

        raise NotImplementedError("`preprocessing` has to be overriden.")

    def _post_processing_wrapper(self, outputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """Post-processing wrapper."""

        outputs_is_nested = tf.nest.is_nested(outputs)
        outputs = tuple(outputs) if outputs_is_nested else (outputs,)
        outputs = tf.nest.map_structure(
            lambda output: self.post_processing(output), outputs
        )

        return outputs[0] if len(outputs) == 1 else outputs

    def post_processing(self, output: TENSOR_TYPE) -> None:
        """Public API to apply post-processing logics to your outputs data."""

        raise NotImplementedError("`post_processing` has to be overriden.")

    def init_params(
        self, inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]], **kwargs: dict
    ) -> None:
        """Public API to initialize parameters before `build` method."""

        raise NotImplementedError("`init_params` has to be overriden.")


class QuantileLayer(BaseLayer):
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
        self, apply_quantiles_transpose: bool = False, *args: list, **kwargs: dict
    ) -> None:

        super().__init__(*args, **kwargs)

        self.apply_quantiles_transpose = apply_quantiles_transpose
        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]

    def _set_quantiles(
        self,
        value: List[List[float]],
        additional_shapes: Union[None, List[List[int]]] = None,
        n_quantiles: Union[None, List[List[int]]] = None,
    ) -> None:
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self._built = False
        self._has_quantiles = True
        self._quantiles = value
        self._additional_shapes = additional_shapes or [
            [len(q)] for q in self.quantiles
        ]
        self._n_quantiles = n_quantiles or self._additional_shapes.copy()

    def preprocessing(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """No preprocessing for `QuantileModel`"""
        return inputs

    def post_processing(self, outputs: TENSOR_TYPE, **kwargs: dict) -> TENSOR_TYPE:
        """Convert the outputs to `QuantileTensor` and apply transpose operation.

        The quantiles dimension is put to the last dimension to fit with keras norms.
        There is a difference with its equivalent Model implementation, we can't check with losses
        if they have a quantile attribute hence `apply_quantiles_transpose` is set to False by default
        and if you need to implement a layer with transpose operation you have to set it to True.

        The only check used is to ensure that quantile dimension is present in the outputs tensors.
        """

        if self.apply_quantiles_transpose:
            if self._check_quantiles_requirements(outputs, **kwargs):
                outputs = transpose_first_to_last(outputs)
                if outputs.shape[-1] == 1:
                    outputs = tf.squeeze(outputs, axis=-1)
                return QuantileTensor(outputs, quantiles=True)
            return QuantileTensor(outputs, quantiles=False)
        return outputs

    def _check_quantiles_requirements(
        self, outputs: TENSOR_TYPE, **kwargs: dict
    ) -> bool:
        """Check if the requirements are valids.
        """

        if self.has_quantiles:
            check_quantiles_in_outputs = self._check_quantiles_in_outputs(outputs)
            if check_quantiles_in_outputs:
                return True
        return False

    def _check_quantiles_in_outputs(self, outputs: TENSOR_TYPE) -> bool:
        """Return True if the outputs contains a `quantiles` dimension."""

        # TODO: find an other way to find if an outputs contains quantiles dimension

        return any(
            s == outputs.shape[: len(s)] for s in self._additional_shapes
        )  # or self._additional_shapes

    def init_params(
        self, inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]], **kwargs: dict
    ) -> None:
        pass

    def get_additional_shapes(self, index: int) -> Union[List[int], List[None]]:
        """Return the shape to add to your layers.
        
        How works this method: 
        If you defined two :class:`autopycoin.losses.QuantileLossError`
        in your model with two differents `quantiles` attribute
        for your two outputs tensors then index=0 will access the shape associated 
        to the first `quantiles` attribute.
        Else it gives an empty list.
        """

        try:
            return self._additional_shapes[index]
        except IndexError:
            return []

    def init_params(
        self, inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]], **kwargs: dict
    ) -> None:
        pass

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
        self, apply_multivariate_transpose: bool = False, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self.apply_multivariate_transpose = apply_multivariate_transpose
        self._init_multivariates_params = False

        self._n_variates = []
        self._is_multivariate = False

    def preprocessing(
        self, inputs: TENSOR_TYPE
    ) -> Union[tf.Tensor, tf.Variable, UnivariateTensor]:
        """Init the multivariates attributes and transpose the `nvariates` dimension in first position."""

        if self.apply_multivariate_transpose and self.is_multivariate:
            return tf.nest.map_structure(transpose_last_to_first, inputs)
        return inputs

    def post_processing(self, outputs: TENSOR_TYPE, **kwargs: dict) -> TENSOR_TYPE:
        outputs = super().post_processing(outputs, **kwargs)
        if self.apply_multivariate_transpose:
            if self.is_multivariate:
                outputs = tf.nest.map_structure(
                    lambda outputs: transpose_first_to_second_last(outputs)
                    if outputs.quantiles
                    else transpose_first_to_last(outputs),
                    outputs,
                )
                return tf.nest.map_structure(
                    convert_to_univariate_tensor(multivariates=True), outputs
                )
            return tf.nest.map_structure(
                convert_to_univariate_tensor(multivariates=False), outputs
            )
        return outputs

    def init_params(
        self,
        inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]],
        n_variates: Union[None, List[Union[None, int]]] = None,
        is_multivariate: Union[None, bool] = None,
        additional_shapes: Union[None, List[List[int]]] = None,
    ) -> None:
        """Initialize attributes related to univariate model.
        
        It is called before `build`.
        Three steps are done:
        - Filter the first shape in case of multiple inputs tensors.
        - Initialize attributes: `is_multivariate`, `n_variates`.
        - Add the n_variates dimension to `additional_shape`.
        """

        if not self._init_multivariates_params:
            if isinstance(inputs_shape, (tuple, list)):
                inputs_shape = inputs_shape[0]

            self._init_multivariates_params = True
            self._set_is_multivariate(inputs_shape, is_multivariate)
            self._set_n_variates(inputs_shape, n_variates)
            self._extend_additional_shape(additional_shapes)

    def _set_is_multivariate(
        self, inputs_shape: tf.TensorShape, is_multivariate: Union[None, bool] = None
    ) -> None:
        """Initiate `is_multivariate` attribute"""

        self._is_multivariate = is_multivariate or bool(inputs_shape.rank > 2)

    def _set_n_variates(
        self,
        inputs_shape: tf.TensorShape,
        n_variates: Union[None, List[Union[None, int]]] = None,
    ) -> None:
        """Initiate `n_variates` attribute"""

        if self.is_multivariate:
            self._n_variates = convert_to_list(n_variates or inputs_shape[-1])

    def _extend_additional_shape(
        self, additional_shapes: Union[None, List[List[int]]] = None
    ) -> None:
        self._additional_shapes = additional_shapes or [
            s + self.n_variates for s in self._additional_shapes
        ]

    @property
    def is_multivariate(self) -> bool:
        return self._is_multivariate

    @property
    def n_variates(self) -> List[Union[None, int]]:
        return self._n_variates


def convert_to_univariate_tensor(multivariates):
    def fn(tensor):
        return UnivariateTensor(values=tensor, multivariates=multivariates)

    return fn
