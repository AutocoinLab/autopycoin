"""This file defines the strategy to adopt."""

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

import numpy as np
from .mlcore import CoreModel 
from ..utils.data_adapter import unpack_timeseries


class OneShot(CoreModel):
    def __init__(self, model, n_steps, **kwargs):
            
        super().__init__(model=model, **kwargs)
        
        self._n_steps = n_steps
    
    def fit(self,
            x=None,
            y=None,
            callbacks=None,
            **kwargs):
        
        for _, labels in self._extract_sample(x): 
            self._len_labels = labels.shape[-1] # Used in predict
        
        super().fit(x=x,
                    y=y,
                    callbacks=callbacks,
                    **kwargs)
        
    def predict(self, x):
        """ Predict output according to x.
        """
            
        ys = super().predict(x).reshape((-1, self._n_steps, self._len_labels))
        
        return ys
    
    
    
class AutoRegressive(CoreModel):
    def __init__(self,model, n_steps, labels_in_inputs_indices, **kwargs):
        
        super().__init__(model=model, **kwargs)
        
        if isinstance(labels_in_inputs_indices, dict):
            self.labels_in_inputs_indices = list(labels_in_inputs_indices.values())
        elif isinstance(labels_in_inputs_indices, list):
            self.labels_in_inputs_indices = labels_in_inputs_indices
        else:
            raise ValueError(
                      f"The label_in_inputs_idx parameter need to be a dictionnary, list or `None`. Got {labels_in_inputs_indices}"
                      " instead.")
            
        self._n_steps = n_steps
        
    def predict_format(self, inputs, labels):
        
        if len(inputs) == 4:
            inputs, _, date_inputs, date_labels = inputs
        else:
            inputs, date_inputs, date_labels = inputs
        
        # Workout number of batch
        assert self._n_steps < inputs.shape[0]
        
        self._n_batchs = inputs.shape[0] - self._n_steps + 1 # 1 represents the Shift
        
        print(inputs.shape[0])
        print(self._n_batchs)
            
        # From instances create batches of size n_steps
        """for idx in tf.range(self._n_batchs):
            print(inputs[idx:-(self._n_batchs - 1 - idx), :])"""
        inputs = tf.stack([inputs[idx:self._n_batchs + 1 + idx, :] for idx in tf.range(self._n_batchs)])
        date_inputs = tf.stack([tf.reshape(date_inputs[idx:idx+self._n_steps, :], shape=(-1,))
                                for idx in tf.range(self._n_batchs)])
        date_labels = tf.stack([tf.reshape(date_labels[idx:idx+self._n_steps, :], shape=(-1,)) 
                                for idx in tf.range(self._n_batchs)])
        labels = tf.stack([labels[idx:idx+self._n_steps, :] for idx in tf.range(self._n_batchs)]) 
        
        return (inputs, date_inputs, date_labels), labels
    
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
        x, y, sample_weights = data_adapter.unpack_x_y_sample_weight(data)
        x, y = self.predict_format(x, y)
        x = unpack_timeseries(x)
        return self(x, training=False)
    
    def fit(self,
            x=None,
            y=None,
            callbacks=None,
            **kwargs):
        
        for _, labels in self._extract_sample(x): 
            self._len_labels = labels.shape[-1] # Used to create output tensor in call function
            
        super().fit(x=x,
                    y=y,
                    callbacks=callbacks,
                    **kwargs)
        
    def call(self, inputs, training=False):
        """Inference of the model.
        This method is used for prediction and evaluation of a trained model.

        Args:
          x: Numpy array, tensor or tensorflow dataset with tensors of shape (timesteps, variables).
             Ground truth inputs. The model assumes the input matrix is shape (timesteps, variables) 
             and performs transformations in order to match auto regressive hypotheses:
              - Based on n_steps it will create batch of inputs (xs) and initialize outputs (ys).
              - At each step, it takes the previous predictions and insert them in x in order to avoid data leakage.
              
        Returns:
          ys: Numpy array of shape (batch_size, variables * timesteps).
              Predictions.
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
            

        predictions = np.zeros((self._n_batchs, self._n_steps, self._len_labels), dtype='float64')

        for idx in tf.range(self._n_steps): 
            x = inputs[:, idx].numpy() # Select all batches and one step
            if idx != 0 and self.labels_in_inputs_indices is not None:
                x[:, self.labels_in_inputs_indices] = predictions[:, idx-1]

            predictions[:, idx] = self._model.predict(x).reshape(-1, self._len_labels)

        return predictions

