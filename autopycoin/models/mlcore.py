from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from functools import partial  # pylint: disable=g-importing-member
import inspect
import os
import tempfile
from typing import Optional, List, Dict, Any, Union, Text, Tuple, NamedTuple, Set
import uuid

from ..utils.data_adapter import unpack_timeseries
from absl import logging
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import MapDataset
from tensorflow.python.keras.engine import data_adapter

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
losses = tf.keras.losses
backend = tf.keras.backend


class CoreModel(models.Model):
    """Keras Model V2 wrapper around numpy based model like sklearn model.
      Basic usage example:
      ```python
      ```
      Using this model has some caveats:
        * Models are not Neural Networks. Feature preprocessing that
          are beneficial to neural network (normalization, one-hot encoding) can be
          detrimental to other models.
        * During training, the entire dataset is loaded in memory (in an efficient
          representation). In case of large datasets (>100M examples), it is
          recommended to randomly downsample the examples.
        * The model trains for exactly one epoch. The core of the training
          computation is done at the end of the first epoch. The console will show
          training logs (including validations losses and feature statistics).
        * The model cannot make predictions before the training is done. Applying
          the model before training will raise an error. During training Keras
          evaluation will be invalid (the model always returns zero).
      Attributes:
        model: Numpy based Model which define a fit and predict methods.
        preprocessing: Functional keras model or @tf.function to apply on the input
          feature before the model to train. This preprocessing model can consume
          and return tensors, list of tensors or dictionary of tensors. If
          specified, the model only "sees" the output of the preprocessing (and not
          the raw input). Can be used to prepare the features or to stack multiple
          models on top of each other. Unlike preprocessing done in the tf.dataset,
          the operation in "preprocessing" are serialized with the model.
        verbose: If true, displays information about the training.
    """      

    def __init__(self, 
                 model = None,
                 verbose: Optional[bool] = True,
                 preprocessing = None) -> None:
        
        super(CoreModel, self).__init__()
    
        self._preprocessing = preprocessing
        self._verbose = verbose

        # Internal, indicates whether the first evaluation during training,
        # triggered by providing validation data, should trigger the training
        # itself.
        self._train_on_evaluate: bool = False

        # True iif. the model is trained.
        self._is_trained = tf.Variable(False, trainable=False, name="is_trained")

        # The following fields contain the trained model. They are set during the
        # graph construction and training process.

        # The compiled model.
        self._model = model

        # Textual description of the model.
        self._description: Optional[Text] = None
             
    def predict_step(self, data):
        """The logic for one inference step.
        This method can be overridden to support custom inference logic.
        This method is called by `Model.make_predict_function`.
        This method should contain the mathematical logic for one step of inference.
        This typically includes the forward pass.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_predict_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          The result of one inference step, typically the output of calling the
          `Model` on data.
        """

        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        x = unpack_timeseries(x)
        return self(x, training=False)

    def make_predict_function(self):
        """Prediction of the model (!= evaluation)."""

        def predict_function_not_trained(iterator):
            """Prediction of a non-trained model. Returns "zeros"."""

            data = next(iterator)
            data = _expand_1d(data)
            x, _, _ = unpack_x_y_sample_weight(data)
            batch_size = _batch_size(x)
            return tf.zeros([batch_size, 1])

        def predict_function_trained(iterator, model):
            """Prediction of a trained model.
            The only difference with "super.make_predict_function()" is that
            "self.predict_function" is not set and that the "distribute_strategy"
            is not used.
            Args:
              iterator: Iterator over the dataset.
              model: Model object.
            Returns:
              Model predictions.
            """

            def run_step(data):
                outputs = model.predict_step(data)
                with tf.control_dependencies(_minimum_control_deps(outputs)):
                    model._predict_counter.assign_add(1)  # pylint:disable=protected-access
                return outputs
            data = next(iterator)
            return run_step(data)

        if self._is_trained:
            return partial(predict_function_trained, model=self)
        else:
            return predict_function_not_trained
        
    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        
        x = x.map(self.numpy_format) # x needs to be transformed in order to feed numpy based model 
        
        return super().predict(x,
                               batch_size=None,
                               verbose=0,
                               steps=None,
                               callbacks=None,
                               max_queue_size=10,
                               workers=1,
                               use_multiprocessing=False)

    def make_test_function(self):
        """Predictions for evaluation."""

        def test_function_not_trained(iterator):
            """Evaluation of a non-trained model."""

            next(iterator)
            return {}

        def test_function_trained(iterator, model):
            """Evaluation of a trained model.
            The only difference with "super.make_test_function()" is that
            "self.test_function" is not set.
            Args:
              iterator: Iterator over dataset.
              model: Model object.
            Returns:
              Evaluation metrics.
            """

            def run_step(data):
                outputs = model.test_step(data)
                with tf.control_dependencies(_minimum_control_deps(outputs)):
                    model._test_counter.assign_add(1)  # pylint:disable=protected-access
                return outputs

            data = next(iterator)
            return run_step(data)

        if self._is_trained:
            return partial(test_function_trained, model=self)
        else:
            return test_function_not_trained

    def call(self, inputs, training=False):
        """Inference of the model.
        This method is used for prediction and evaluation of a trained model.
        Args:
          inputs: Input tensors.
          training: Is the model being trained. Always False.
        Returns:
          Model predictions.
        """

        del training

        assert self._model is not None

        if self._preprocessing is not None:
            inputs = self._preprocessing(inputs)

        if isinstance(inputs, tf.Tensor):
            # Native format
            pass
        elif isinstance(inputs, dict):
            # Native format
            pass
        elif isinstance(inputs, list):
            # Note: The name of a tensor (value.name) can change between the training
            # and the inference.
            inputs = {str(idx): value for idx, value in enumerate(inputs)}          
        else:
            error_msg = ("Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                     "or `(x, y, sample_weight)`, found: {}").format(inputs)
            
        # Apply the model.
        predictions = self._model.predict(inputs)
        return predictions

    def train_step(self, data):
        """Collects training examples."""

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        if self._verbose:
            logging.info("Collect training examples.\nFeatures: %s\nLabel: %s",
                   x, y)

        if self._preprocessing is not None:
            x = self._preprocessing(x)
        if self._verbose:
            logging.info("Applying preprocessing on inputs. Result: %s", x)

        if isinstance(x, tf.data.Dataset):
            # Native format
            pass
        elif isinstance(x, dict):
            # Native format
            pass
        elif isinstance(x, list):
            # Note: The name of a tensor (value.name) can change between the training
            # and the inference.
            x = {str(idx): value for idx, value in enumerate(x)}

        if not isinstance(y, tf.Tensor):
            raise ValueError(
          f"The training label tensor is expected to be a tensor. Got {y}"
          " instead.")
           
        self._train_x = unpack_timeseries(x)
        self._train_y = y

        # Not metrics are returned during the collection of training examples.
        return {}

    def compile(self, metrics=None):
        """Configure the model for training.
        Unlike for most Keras model, calling "compile" is optional before calling
        "fit".
        Args:
          metrics: Metrics to report during training.
        Raises:
          ValueError: Invalid arguments.
        """

        super(CoreModel, self).compile(run_eagerly=True, metrics=metrics)

    def fit(self,
            x=None,
            y=None,
            callbacks=None,
            **kwargs) -> tf.keras.callbacks.History:
        """Trains the model.
        The following dataset formats are supported:
          1. "x" is a tf.data.Dataset containing a tuple "(features, labels)".
             "features" is list of tensors. "labels" is a tensor.
          2. "x" is a list of tensors containing
             the input features. "y" is a tensor.
          3. "x" is a list of numpy-arrays containing the input features. "y" is a numpy-array.
        Pandas Dataframe can be consumed with "WindowGenerator":
          dataset = pandas.Dataframe(...)
          w = WindowGenerator(data=dataset, input_width=window, label_width=forecast, 
                              shift=shift, test_size=test_size, 
                              valid_size=valid_size, batch_size=batch_size,
                              input_columns=input_columns,
                              known_columns=known_columns, 
                              label_columns=label_columns,
                              date_columns=date_columns)
          m = OneShot(model, n_steps=n_steps)
          m.compile()
          m.fit(w.train)
        Args:
          x: Training dataset (See details above for the supported formats).
          y: Label of the training dataset. Only used if "x" does not contains the
            labels.
          callbacks: Callbacks triggered during the training.
          **kwargs: Arguments passed to the core keras model's fit.
        Returns:
          A `History` object. Its `History.history` attribute is not yet
          implemented, and will return empty.
          All other fields are filled as usual for `Keras.Model.fit()`.
        """
        
        x = x.map(self.numpy_format)
            
        # Check for a Pandas Dataframe without injecting a dependency.
        if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
            raise ValueError(
                "`fit` cannot consume Pandas' dataframes directly. Instead, use the "
                "`pd_dataframe_to_tf_dataset` utility function. For example: "
                "`model.fit(tfdf.keras.pd_dataframe_to_tf_dataset(train_dataframe, "
                "label=\"label_column\"))")

        # Call "compile" if the user forgot to do so.
        if not self._is_compiled:
            self.compile()

        if "epochs" in kwargs:
            if kwargs["epochs"] != 1:
                raise ValueError("all decision forests algorithms train with only 1 " +
                         "epoch, epochs={} given".format(kwargs["epochs"]))
            del kwargs["epochs"]  # Not needed since we force it to 1 below.

        # This callback will trigger the training at the end of the first epoch.
        callbacks = [_TrainerCallBack(self)] + (callbacks if callbacks else [])

        # We want the model trained before any evaluation is done at the
        # end of the epoch. This may fail in case any of the `on_train_batch_*`
        # callbacks calls `evaluate()` before the end of the 1st epoch.
        self._train_on_evaluate = True

        try:
            history = super(CoreModel, self).fit(
              x=x, y=y, epochs=1, callbacks=callbacks, **kwargs)
        finally:
            self._train_on_evaluate = False

        self._build(x)

        return history

    # TO DO
    def save(self, filepath: str, overwrite: Optional[bool] = True, **kwargs):
        """Saves the model as a TensorFlow SavedModel.
        The exported SavedModel contains a standalone Yggdrasil Decision Forests
        model in the "assets" sub-directory. The Yggdrasil model can be used
        directly using the Yggdrasil API. However, this model does not contain the
        "preprocessing" layer (if any).
        Args:
          filepath: Path to the output model.
          overwrite: If true, override an already existing model. If false, raise an
            error if a model already exist.
          **kwargs: Arguments passed to the core keras model's save.
        """
        
        if tf.io.gfile.exists(os.path.join(filepath, "saved_model.pb")):
            if overwrite:
                tf.io.gfile.rmtree(filepath)
        else:
            raise ValueError(
            f"A model already exist as {filepath}. Use an empty directory "
            "or set overwrite=True")

        super(CoreModel, self).save(
            filepath=filepath, overwrite=overwrite, **kwargs)

    # TO DO
    def evaluate(self, *args, **kwargs):
        """Returns the loss value & metrics values for the model.
        See details on `keras.Model.evaluate`.
        Args:
          *args: Passed to `keras.Model.evaluate`.
          **kwargs: Passed to `keras.Model.evaluate`.  Scalar test loss (if the
            model has a single output and no metrics) or list of scalars (if the
            model has multiple outputs and/or metrics). See details in
            `keras.Model.evaluate`.
        """
        if self._train_on_evaluate:
            if not self._is_trained.numpy():
                self._train_model()
        else:
            raise ValueError(
            "evaluate() requested training of an already trained model -- "
            "did you call `Model.evaluate` from a `on_train_batch*` callback ?"
            "this is not yet supported in Decision Forests models, where one "
            "can only evaluate after the first epoch is finished and the "
            "model trained")
        return super(CoreModel, self).evaluate(*args, **kwargs)

    # TO DO
    def summary(self, line_length=None, positions=None, print_fn=None):
        """Shows information about the model."""

        super(CoreModel, self).summary(
            line_length=line_length, positions=positions, print_fn=print_fn)

        if print_fn is None:
            print_fn = print

    # TO DO
    def _extract_sample(self, x):
        """Extracts a sample (e.g.
        batch, row) from the training dataset.
        Returns None is the sample cannot be extracted.
        Args:
          x: Training dataset in the same format as "fit".
        Returns:
          A sample.
        """

        if isinstance(x, tf.data.Dataset):
            return x.take(1)

        try:
            # Work for numpy array and TensorFlow Tensors.
            return tf.nest.map_structure(lambda v: v[0:1], x)
        except Exception:  # pylint: disable=broad-except
            pass

        try:
            # Works for list of primitives.
            if isinstance(x, list) and isinstance(x[0],
                                                (int, float, str, bytes, bool)):
                return x[0:1]
        except Exception:  # pylint: disable=broad-except
            pass

        logging.warning("Dataset sampling not implemented for %s", x)
        return None

    def _build(self, x):
        """Build the internal graph similarly as "build" for classical Keras models.
        Compared to the classical build, supports features with dtypes != float32.
        Args:
          x: Training dataset in the same format as "fit".
        """

        # Note: Build does not support dtypes other than float32.
        super(CoreModel, self).build([])

    def _train_model(self):
        """Effectively train the model."""

        if self._train_x is None:
            raise Exception("The training data was not built.")

        # Train the model.
        self._model.fit(self._train_x, self._train_y)
        
        self._is_trained.assign(True)
        
    def numpy_format(self, inputs, labels):
        """Flatten the inputs in order to feed numpy based model."""
        
        inputs, known, date_inputs, date_labels =  inputs

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs = tf.reshape(inputs, shape=(-1, tf.reduce_prod(inputs.shape[1:])))

        if known is not None:
            known = tf.reshape(known, shape=(-1, tf.reduce_prod(known.shape[1:])))
            inputs = tf.concat((inputs, known), axis=-1)

        labels = tf.reshape(labels, shape=(-1, tf.reduce_prod(labels.shape[1:])))
        date_inputs = tf.reshape(date_inputs, shape=(-1, tf.reduce_prod(date_inputs.shape[1:])))
        date_labels = tf.reshape(date_labels, shape=(-1, tf.reduce_prod(date_labels.shape[1:])))

        return (inputs, date_inputs, date_labels), labels


class _TrainerCallBack(tf.keras.callbacks.Callback):
    """Callback that trains the model at the end of the first epoch."""

    def __init__(self, model: CoreModel):
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        del logs
        if epoch == 0 and not self._model._is_trained.numpy():  # pylint:disable=protected-access
            self._model._train_model()  # pylint:disable=protected-access
        
        # After this the model is trained, and evaluations shouldn't attempt
        # to retrain.
        self._model._train_on_evaluate = False  # pylint:disable=protected-access


def _batch_size(inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
    """Gets the batch size of a tensor or dictionary of tensors.
      Assumes that all the tensors have the same batchsize.
      Args:
        inputs: Dict of tensors.
      Returns:
        The batch size.
      Raises:
        ValueError: Invalid arguments.
    """

    if isinstance(inputs, dict):
        for v in inputs.values():
            return tf.shape(v)[0]
        raise ValueError("Empty input")
    else:
        return tf.shape(inputs)[0]
    
    
def _minimum_control_deps(outputs):
    """Returns the minimum control dependencies to ensure step succeeded.
      This function is a strict copy of the function of the same name in the keras
      private API:
      third_party/tensorflow/python/keras/engine/training.py
    """

    if tf.executing_eagerly():
        return []  # Control dependencies not needed.
    outputs = tf.nest.flatten(outputs, expand_composites=True)
    for out in outputs:
        # Variables can't be control dependencies.
        if not isinstance(out, tf.Variable):
            return [out]  # Return first Tensor or Op from outputs.
    return []  # No viable Tensor or Op to use for control deps.