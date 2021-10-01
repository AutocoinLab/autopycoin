"""This file defines the WindowGenerator model.
"""

import numpy as np
import tensorflow as tf


class WindowGenerator:
    """
    Transform a time series dataset into an usable format.

    Parameters
    ----------
        data: Dataframe of shape (timesteps, variables)
            The time series dataframe.
        input_width: int
            The number of historical time steps to use in the model.
        label_width: int
            the number of time steps to forecast.
        shift: int
            Compute the shift between inputs variables and labels variables.
        valid_size: int
            The Number of examples in the validation set.
        test_size: int
            The Number of examples in the test set.
        batch_size: int, default to None.
            The number of examples per batch. If None, then batch_size = len(data)
        input_columns: list[str]
            The input columns names, default to None.
        known_columns: list[str]
            The known columns names, default to None.
        label_columns: list[str]
            The label columns names, default to None.
        date_columns: list[str]
            The date columns names, default to None.
            Date columns will be cast to string and join by
            '-' delimiter to be used as xticks.
        preprocessing : Callable
            Preprocessing function to use on the data. 
            This function will to take input of shape ((inputs, known, date_inputs, date_labels), labels)

    Attributes
    ----------
        input_width : int
        label_width : int
        shift : int
        total_window_size : int
        input_slice : slice
        input_indices : array
        label_start : int
        label_slice : slice
        label_indices : array
        valid_size : int
        test_size : int
        batch_size : int
        train : dataframe
        valid : dataframe
        test : dataframe

    Notes
    -----
    output shape:
    tuple ((inputs, known, date_inputs, date_labels), labels)
    with tensors of shape: (quantiles, batch_size, ..., units).
    """

    def __init__(
        self,
        data,
        input_width,
        label_width,
        shift,
        test_size,
        valid_size,
        batch_size=None,
        input_columns=None,
        known_columns=None,
        label_columns=None,
        date_columns=None,
        preprocessing=None,
    ):

        # Work out the window parameters.
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.shift = int(shift)

        self.total_window_size = self.input_width + self.shift

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

        self.valid_size = int(valid_size)
        self.test_size = int(test_size)

        self.batch_size = batch_size

        test_start = data.shape[0] - (self.total_window_size + self.test_size - 1)
        valid_start = test_start - (self.total_window_size + self.valid_size - 2)

        # Filter the columns
        self.known_columns = known_columns
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.date_columns = date_columns

        data = data.loc[
            :,
            set(
                (input_columns or [])
                + (known_columns or [])
                + (date_columns or [])
                + (label_columns or [])
            ),
        ]

        # Work out the datasets.
        self.train_ds = data
        if (self.test_size + self.valid_size) != 0:
            self.train_ds = data.iloc[: (valid_start + self.input_width)]

        self.valid_ds = None
        if valid_size != 0:
            self.valid_ds = data.iloc[valid_start : (test_start + self.input_width)]

        self.test_ds = None
        if test_size != 0:
            self.test_ds = data.iloc[test_start:]

        # Work out the column indices.
        # label indices according to the label dataset
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        # Column indices according to the input dataset
        self.inputs_columns_indices = {
            name: i for i, name in enumerate(input_columns)
        }

        # label indices according to the input dataset
        self.labels_in_inputs_indices = {
            key: value
            for key, value in self.inputs_columns_indices.items()
            if key in label_columns
        }

        # Columns indices according to train_ds
        self.column_indices = {
            name: i for i, name in enumerate(self.train_ds)
        }  

        self._preprocessing = preprocessing

    def make_dataset(self, data):
        """
        Compute the tensorflow dataset object.
        
        Parameters
        ----------
            data: dataframe or array or tensor of shape (timestep, variables)
                The time series dataset.

        Returns
        -------
            ds: Tensorflow dataset.
                The dataset that can be used in keras model.
        """

        batch_size = self.batch_size

        if self.batch_size is None:
            batch_size = len(data)

        # Necessary because ML model need all values
        data = tf.convert_to_tensor(data, dtype=tf.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=batch_size,
        )

        ds = ds.map(self.split_window)

        if self._preprocessing is not None:
            ds = ds.map(self._preprocessing)

        return ds

    def split_window(self, features):
        """Compute the window split.
        
        Parameters
        ----------
            features: tensor of shape (Batch_size, timestep, variables)
                The window defined by timeseries_dataset_from_array class.

        Returns
        -------
            inputs: tensor of shape (batch_size, input_width, variables)
                The input variables.
            known: tensor of shape (batch_size, label_width, variables)
                The known variables. class of variables whose values are known
                in advance or estimated as dates or temperatures.
            date_inputs: tensor of shape (batch_size, input_width, 1)
                Input dates, default to a tensor generated by `tf.range` of shape
                (input_width, 1).
            date_labels: tensor of shape (batch_size, label_width, 1)
                label dates, default to a tensor generated by `tf.range` of shape
                (label_width, 1).
            labels: tensor of shape (batch_size, label_width, variables)
                The Output variables.
        """

        # Workout Date
        if self.date_columns is not None:
            date = tf.stack(
                [
                    features[:, :, self.column_indices[name]]
                    for name in self.date_columns
                ],
                axis=-1,
            )
            date.set_shape([None, self.total_window_size, None])
            date = tf.cast(date, tf.int32)
            date = tf.strings.as_string(date)
            date = tf.strings.reduce_join(date, separator="-", axis=2)
        else:
            date = tf.reshape(tf.range(tf.shape(features)[1]), shape=(-1, 1))

        date_inputs = date[:, self.input_slice]
        date_labels = date[:, self.label_slice]

        # Workout Known inputs
        if self.known_columns is not None:
            known = tf.stack(
                [
                    features[:, self.label_slice, self.column_indices[name]]
                    for name in self.known_columns
                ],
                axis=-1,
            )
            known.set_shape([None, self.label_width, None])
        else:
            known = None

        # Workout inputs and labels
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]

        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
            axis=-1,
        )
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1,
        )

        inputs.set_shape([None, self.input_width, len(self.input_columns)])
        labels.set_shape([None, self.label_width, len(self.label_columns)])

        return (inputs, known, date_inputs, date_labels), labels

    @property
    def train(self):
        """Build the train dataset."""
        return self.make_dataset(self.train_ds)

    @property
    def valid(self):
        """Build the valid dataset."""
        return self.make_dataset(self.valid_ds)

    @property
    def test(self):
        """Build the test dataset."""
        return self.make_dataset(self.test_ds)

    def forecast(self, data):
        """Build the production dataset.

        parameters
        ----------
            data: DataFrame, array or tensor of shape (steps, variables)
                data needs to contains the inputs and outputs where labels are set
                to zero.
        """

        data = data.loc[:, self.column_indices]

        return self.make_dataset(data)

    def __repr__(self):
        """Display some explanations."""

        example = iter(self.train)
        ((inputs, known, date_inputs, date_labels), labels) = example.get_next()

        return f"""Generator starting... \n

            Input columns are : {self.input_columns}
            Known colulns are : {self.known_columns}
            Labels colulmns are : {self.label_columns}
            date colulns are : {self.date_columns} \n

            Associated indices to each column are: {self.column_indices} \n

            Parameter remainder:\n
            - input_width : {self.input_width}
            - label_width : {self.label_width}
            - shift : {self.shift}
            - test_size : {self.test_size}
            - valid_size : {self.valid_size}
            - batch_size : {self.batch_size} \n

            The train set becomes : \n {self.train_ds} \n
            The validation set becomes : \n {self.valid_ds} \n
            The test set becomes : \n {self.test_ds} \n

            A split exemple: \n
                Inputs : \n {inputs}
                Known inputs : \n {known}
                Input dates : \n {date_inputs}
                Label dates : \n {date_labels}
                Labels : \n {labels}

                \n Label indices \n {self.label_indices}
                \n Input indices \n {self.input_indices}
                """
