"""Defines the pool model."""


from typing import Union, List, Callable, Optional, Tuple
import tensorflow as tf
from keras.engine import data_adapter

from ..utils.data_utils import convert_to_list

# TODO: AutopycoinBaseClass

# TODO: finish doc and unit testing.
class BasePool(tf.keras.Model):
    """Tensorflow model defining a pool of models.

    For now, it implements the bagging method.
    This model needs to be overriden.

    `preprocessing_x`, `preprocessing_y` and `postprocessing_y` are 
    applied to adapt the model.

    Parameters
    ----------
    label_width : int
        Width of the targets.
        It can be not defined if `nbeats_model` is a list of NBEATS instances.
        Default to None.
    n_models : int
        Number of models inside the pool. This is used only when callables are
        provided.
    models : list[callable[tf.keras.Model]] or List[tf.keras.Model]
        Can be a list of callables which create model or a list of instantiate models.
        The callables have to let only `label_width` parameter free.
    fn_agg : Callable
        Function of aggregation which takes an parameter axis.
        It aggregates the model's outputs tensor. The outputs are aggregated only if
        all the shapes are similar else return the structure.
        Default to mean.
    seed : int
        Used in combination with tf.random.set_seed to create a
        reproducible sequence of tensors across multiple calls.

    Attributes
    ----------
    fn_agg : Callable
    n_models : int
    models : List[NBEATS]
    label_width : int
    """

    def __init__(
        self,
        label_width: int,
        n_models: Union[None, int],
        models: List[Union[tf.keras.Model, Callable[..., tf.keras.Model]]],
        fn_agg: Callable[..., tf.Tensor],
        model_distribution: List[int] = None,
        seed: Optional[int] = None,
        **kwargs: dict,
    ):

        super().__init__(**kwargs)

        # Reproducible instance
        tf.random.set_seed(seed)
        self.seed = seed

        self._n_models = n_models
        self._label_width = label_width

        # models init
        self._init_models(models, model_distribution=model_distribution, **kwargs)

        # Layer definition and function to aggregate the multiple outputs
        self._fn_agg = fn_agg

    def _init_models(
        self,
        models: List[Union[tf.keras.Model, Callable]],
        model_distribution: List[int],
        **kwargs: dict,
    ) -> None:
        """Initialize models by picking randomly models or by using instances."""

        models = convert_to_list(models)

        if any(isinstance(model, tf.keras.Model) for model in models):
            # We modify n_models
            self._n_models = len(models)

            # Check only if instances are provided
            self.checks(
                (model for model in models if isinstance(model, tf.keras.Model))
            )
            self._models = self._init_callable_models(
                models, distribution=list(range(self.n_models)), **kwargs
            )

        else:
            distribution = (
                model_distribution
                or tf.random.uniform(
                    (self.n_models,), 0, len(models), dtype=tf.int32, seed=self.seed
                )
                .numpy()
                .tolist()
            )

            assert self.label_width and self.n_models, (
                f"When `models` are callable, `label_width` and `n_models` have to be integers. "
                f"Got label_width: {self.label_width} and n_models: {self.n_models} "
                f"and models: {models}."
            )

            self._models = self._init_callable_models(
                models, distribution=distribution, **kwargs
            )

    def _init_callable_models(
        self, models: List[Callable], distribution: List[int], **kwargs
    ) -> List[tf.keras.Model]:
        """Initialize models from callable."""

        def init(idx):
            if isinstance(models[idx], tf.keras.Model):
                return models[idx]
            elif callable(models[idx]):
                model = models[idx](label_width=self.label_width, **kwargs)
                if not isinstance(model, tf.keras.Model):
                    raise ValueError(
                        f"The callables has to return a tensorflow Model, got type {type(model)}."
                    )
                return model
            raise ValueError(
                f"`model` parameter has to be a list of callable or tf.keras.Model or both. Got {models}."
            )

        return tf.nest.map_structure(init, distribution)

    def checks(self, models):
        """
        
        """
        raise NotImplementedError("You need to override this function.")

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    ):

        """Compiles models one by one for training.

        See tensorflow documentation for more informations.
        """

        # Case 1: nmodels > or < losses -> losses for each models
        # Case 2: nmodels == losses -> one loss for one model
        # case 3: nmodels whit n outputs -> list of list -> case 1 ? case 2 ? -> needs to flatten for the overall model

        loss = convert_to_list(loss)
        loss_weights = convert_to_list(loss_weights)

        if len(loss) != self.n_models:

            loss = self._shuffle(loss)

            loss_weights = self._shuffle(loss_weights)

        for idx in range(self.n_models):

            self.models[idx].compile(
                optimizer=optimizer,
                loss=loss[idx],
                metrics=metrics,
                loss_weights=loss_weights[idx],
                weighted_metrics=weighted_metrics,
                **kwargs,
            )

        pool_loss = tf.nest.flatten(loss)
        pool_loss_weights = tf.nest.flatten(loss_weights)

        super().compile(
            optimizer=optimizer,
            loss=pool_loss,
            metrics=metrics,
            loss_weights=pool_loss_weights,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )

    def _shuffle(self, structure,) -> None:
        """Shuffle losses and pick them randomly.
        """

        structure = convert_to_list(structure)

        n_elements = len(structure)

        element_idx = tf.random.uniform(
            (self.n_models,), 0, n_elements, dtype=tf.int32, seed=self.seed
        )

        # Ensure that all losses are represented
        return structure + [structure[idx] for idx in element_idx[n_elements:]]

    def train_step(self, data: tuple) -> dict:
        """The logic for one training step.

        Apply processing function to x and y and train each model separately.
        The results showed during training are calculated from the pool model.
        See tensorflow documentation for more information.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        masked_x = self.preprocessing_x(x)
        masked_y = self.preprocessing_y(y)

        # Run forward pass.
        for idx in range(self.n_models):
            model = self.models[idx]

            # Compute every model forward pass
            model.train_step((masked_x[idx], masked_y[idx], sample_weight))

        # TODO: replace this step for rapidity
        return self.test_step((x, y, sample_weight))

    def test_step(self, data):
        """The logic for one test step.

        Apply processing function to y and test models together.
        See tensorflow documentation for more information.
        """

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y = self.preprocessing_y(y)

        return super().test_step((x, y, sample_weight))

    def predict_step(self, data):
        """The logic for one predict step.

        Apply post-processing function to y.
        See tensorflow documentation for more information.
        """

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y = super().predict_step((x, y, sample_weight))
        y = self.postprocessing_y(y)

        return y

    def preprocessing_x(
        self,
        x: Union[None, Union[Union[tf.Tensor, tf.data.Dataset], Tuple[tf.Tensor, ...]]],
    ) -> Union[Tuple[None, None], Tuple[Callable, tuple]]:
        "Apply mask inside `train_step` and `predict_step`"

        raise NotImplementedError("You need to implement this function.")

    def preprocessing_y(
        self,
        y: Union[None, Union[Union[tf.Tensor, tf.data.Dataset], Tuple[tf.Tensor, ...]]],
    ) -> Union[Tuple[None, None], Tuple[Callable, tuple]]:
        "Apply mask inside `train_step` and `test_step`"

        raise NotImplementedError("You need to implement this function.")

    def postprocessing_y(
        self,
        y: Union[None, Union[Union[tf.Tensor, tf.data.Dataset], Tuple[tf.Tensor, ...]]],
    ) -> Union[Tuple[None, None], Tuple[Callable, tuple]]:
        "Apply mask inside `predict_step`"

        raise NotImplementedError("You need to implement this function.")

    @property
    def fn_agg(self) -> Callable:
        """Return the aggregation function."""
        return self._fn_agg

    @property
    def n_models(self) -> int:
        """Return the `n_models` parameter."""

        return self._n_models

    @property
    def models(self) -> list:
        """Return the nbeats pool."""

        return self._models

    @property
    def label_width(self) -> int:
        """Return the `label_width` parameter."""

        return self._label_width
