"""Defines metrics function"""

from typing import Union, Callable, List, Optional
import tensorflow as tf
from keras.losses import LossFunctionWrapper
from keras.metrics import MeanMetricWrapper, SumOverBatchSizeMetricWrapper

from ..losses import expand_dims, remove_dims

class MetricQuantileDimWrapper(tf.keras.metrics.Metric):

    def __init__(self,
        fn: Union[tf.keras.losses.Loss, LossFunctionWrapper, Callable],
        quantiles: Union[None, List[float]]=None,
        name: Optional[str] = "dim_wrapper",
        *args,
        **kwargs
        ):

        super().__init__(name=name, *args, **kwargs)

        self.fn = fn
        self.quantiles = quantiles
        self._fn_dict = {'expand': expand_dims, 'remove': remove_dims}

        self.preprocess_fn = 'remove'

        if isinstance(self.fn, (MeanMetricWrapper, SumOverBatchSizeMetricWrapper)):
            varnames = self.fn._fn.__code__.co_varnames
        else:
            varnames = self.fn.__code__.co_varnames

        if 'quantiles' in varnames:
            self.preprocess_fn = 'expand'
            kwargs['quantiles'] = self.quantiles
            if not self.quantiles:
                raise ValueError("The model doesn't define a `quantiles` attribute.")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._fn_dict[self.preprocess_fn](y_true, y_pred, self.fn.update_state, sample_weight=sample_weight)

    def result(self):
        return self.fn.result()

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {'fn': self.fn, 'dtype': self.dtype}
