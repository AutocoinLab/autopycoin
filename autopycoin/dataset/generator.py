"""
This file defines the WindowGenerator model.
"""

from typing import Union, Tuple, Mapping, List
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx

from .. import AutopycoinBaseClass


STRATEGIES = ["one_shot", "auto_regressive"]


class WindowGenerator(AutopycoinBaseClass):
    """
    Transform a time series dataset into an usable format.

    Parameters
    ----------
    data : `dataframe of shape (timesteps, variables)`
        The time series dataframe.
    input_width : int
        The number of historical time steps to use in the model.
    label_width : int
        the number of time steps to forecast.
    shift : int
        Compute the shift between inputs variables and labels variables.
    valid_size : int
        The Number of examples in the validation set.
    test_size : int
        The Number of examples in the test set.
    strategy : str
        "one_shot" or "auto_regressive". It defines the inputs shape.
    batch_size : int, `Optional`
        The number of examples per batch. If None, then batch_size = len(data).
        Default to None.
    input_columns : list[str, ...]
        The input columns names.
    label_columns : list[str, ...]
        The label columns names.
    known_columns : list[str, ...] or None, `Optional`
        The known columns names, default to None.
    date_columns : list[str, ...]or None, `Optional`
        The date columns names, default to None.
        Date columns will be cast to string and join by
        '-' delimiter to be used as xticks.
    preprocessing : callable or None, `Optional`
        Preprocessing function to use on the data.
        This function will to take input of shape ((inputs, known, date_inputs, date_labels), labels).
        Default to None.

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
    strategy : str
    batch_size : int
    train : `dataset`
    valid : `dataset`
    test : `dataset`

    Notes
    -----
    *Output shape*:
    tuple ((inputs, known, date_inputs, date_labels), labels)
    with tensors of shape: (batch_size, time_steps, units) or (batch_size, time_steps * units) depending on strategy,
    respectively `auto_regressive` or `one_shot`.

    Examples
    --------
    >>> import pandas as pd
    >>> from autopycoin.data import random_ts
    >>> from autopycoin.dataset import WindowGenerator
    >>> data = random_ts(n_steps=100,
    ...                  trend_degree=2,
    ...                  periods=[10],
    ...                  fourier_orders=[10],
    ...                  trend_mean=0,
    ...                  trend_std=1,
    ...                  seasonality_mean=0,
    ...                  seasonality_std=1,
    ...                  batch_size=1,
    ...                  n_variables=1,
    ...                  noise=True,
    ...                  seed=42)
    >>> data = pd.DataFrame(data[0], columns=['values'])
    >>> w_oneshot = WindowGenerator(data,
    ...                             input_width=3,
    ...                             label_width=2,
    ...                             shift=10,
    ...                             test_size=3,
    ...                             valid_size=2,
    ...                             strategy='one_shot',
    ...                             batch_size=None,
    ...                             input_columns=['values'],
    ...                             label_columns=['values'],
    ...                             known_columns=[],
    ...                             date_columns=[],
    ...                             preprocessing=None)
    >>> w_oneshot.train
    <PrefetchDataset element_spec=((TensorSpec(shape=(None, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 0), dtype=tf.float32, name=None), TensorSpec(shape=(None, 3), dtype=tf.string, name=None), TensorSpec(shape=(None, 2), dtype=tf.string, name=None)), TensorSpec(shape=(None, 2), dtype=tf.float32, name=None))>
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_width: int,
        label_width: int,
        shift: int,
        test_size: int,
        valid_size: int,
        strategy: str,
        input_columns: List[Union[str, None]],
        label_columns: List[Union[str, None]],
        known_columns: List[Union[str, None]] = [],
        date_columns: List[Union[str, None]] = [],
        batch_size: Union[int, None] = None,
        preprocessing: Union[tf.keras.layers.Layer, None] = None,
    ):

        # User Attributes
        self._data = data.copy()

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.valid_size = valid_size
        self.test_size = test_size
        self.batch_size = batch_size

        self.strategy = strategy

        # We separate init functions in order to perfom validation with `__validation` method.
        self._compute_window_parameters(
            input_columns, label_columns, known_columns, date_columns
        )

        self._filtered_data = self._compute_filtered_data(date_columns)

        self._compute_train_valid_test_split()

        self._compute_columns_indices()

        # Preprocessing layers
        self._preprocessing = preprocessing

    def _compute_window_parameters(
        self,
        input_columns: List[Union[str, None]],
        label_columns: List[Union[str, None]],
        known_columns: List[Union[str, None]],
        date_columns: List[Union[str, None]],
    ) -> None:
        """Calculate the window parameters"""
        self.total_window_size = self.input_width + self.shift

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

        self.test_start = self.data.shape[0] - bool(self.test_size) * (
            self.total_window_size + self.test_size - 1
        )
        self.valid_start = self.test_start - bool(self.valid_size) * (
            self.total_window_size + self.valid_size - 2
        )

        # Use columns defined by the `*_columns` parameters
        # Used here and not in `_compute_filtered_data` because we need to
        # verify that columns are inside data fist.
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.known_columns = known_columns
        self.date_columns = date_columns

    def _compute_filtered_data(
        self,
        date_columns: List[Union[str, None]],
    ) -> pd.DataFrame:
        """Return the data filtered by the input, known, label and date columns."""

        filtered_data = self.data.loc[
            :,
            set(
                self.input_columns
                + self.label_columns
                + self.known_columns
                + self.date_columns
            ),
        ]

        # Default creation of a date column
        if not date_columns:
            filtered_data["date"] = range(len(filtered_data))
            self.date_columns = ["date"]

        return filtered_data

    def _compute_train_valid_test_split(
        self,
    ) -> None:
        """Split the data to create train, valid and test datasets"""

        self._train_ds = self.filtered_data.iloc[
            : (self.valid_start + self.input_width)
        ]

        self._valid_ds = self.filtered_data.iloc[
            self.valid_start : (
                self.test_start + self.input_width * bool(self.valid_size)
            )
        ]

        self._test_ds = self.filtered_data.iloc[self.test_start :]

    def _compute_columns_indices(
        self
    ) -> None:
        """Work out columns indices."""
        # labels indices according to the label dataset
        self.label_columns_indices = {
                name: i for i, name in enumerate(self.label_columns)
            }

        # inputs indices according to the input dataset
        self.inputs_columns_indices = {
            name: i for i, name in enumerate(self.input_columns)
        }

        # labels indices according to the input dataset
        self.labels_in_inputs_indices = {
            key: value
            for key, value in self.inputs_columns_indices.items()
            if key in self.label_columns
        }

        # Columns indices according to train_ds
        self.column_indices = {name: i for i, name in enumerate(self.train_ds)}

    def _make_dataset(
        self,
        data: Union[pd.DataFrame, np.array, tf.Tensor],
        batch_size: Union[int, None],
    ) -> tf.data.Dataset:
        """
        Compute the tensorflow dataset object.

        Parameters
        ----------
        data : `dataframe`, array or `tensor of shape (timestep, variables)`
            The time series dataset.
        batch_size : int
            Set up the batch size.

        Returns
        -------
        ds : `dataset`
            The dataset that can be used in keras model.
        """

        if batch_size is None:
            batch_size = len(data)

        # Necessary because ML model need all values
        data = tf.convert_to_tensor(data, dtype=floatx())

        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=batch_size,
        )

        dataset = dataset.map(self._split_window, num_parallel_calls=tf.data.AUTOTUNE)

        if self._preprocessing is not None:
            dataset = dataset.map(
                self._preprocessing, num_parallel_calls=tf.data.AUTOTUNE
            )

        return dataset.prefetch(1)

    def _split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor]:
        """
        Compute the window split.

        Parameters
        ----------
        features : `tensor of shape (Batch_size, timestep, variables)`
            The window defined by timeseries_dataset_from_array class.

        Returns
        -------
        inputs : `tensor of shape (batch_size, input_width, variables)`
            The input variables.
        known : `tensor of shape (batch_size, label_width, variables)`
            The known variables. class of variables whose values are known
            in advance or estimated as dates or temperatures.
        date_inputs : `tensor of shape (batch_size, input_width, 1)`
            Input dates, default to a tensor generated by `tf.range` of shape
            (input_width, 1).
        date_labels : `tensor of shape (batch_size, label_width, 1)`
            label dates, default to a tensor generated by `tf.range` of shape
            (label_width, 1).
        labels : `tensor of shape (batch_size, label_width, variables)`
            The Output variables.
        """
        # We can't use self.batch_size here because the last batch could be truncated
        batch_size = tf.shape(features)[0]

        # Workout Date
        date = tf.stack(
            [features[:, :, self.column_indices[name]] for name in self.date_columns],
            axis=-1,
        )
        date.set_shape([None, self.total_window_size, None])
        date = tf.cast(date, tf.int32)
        date = tf.strings.as_string(date)
        date = tf.strings.reduce_join(date, separator="-", axis=-1)

        date_inputs = date[..., self.input_slice]
        date_labels = date[..., self.label_slice]

        # Workout Known inputs
        if self.known_columns:
            known = tf.stack(
                [
                    features[:, self.label_slice, self.column_indices[name]]
                    for name in self.known_columns
                ],
                axis=-1,
            )
            known.set_shape([None, self.label_width, None])
        else:
            known = tf.repeat(
                tf.constant([[]]), repeats=batch_size, axis=0
            )  # Repeat to fit with batch
            known.set_shape(shape=(None, 0))

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

        # transform this `if` into class strategy
        if self.strategy == "one_shot":
            inputs = tf.reshape(
                inputs, shape=(-1, self.input_width * len(self.input_columns))
            )
            labels = tf.reshape(
                labels, shape=(-1, self.label_width * len(self.label_columns))
            )
            known = tf.reshape(
                known, shape=(-1, self.input_width * len(self.known_columns))
            )
        else:
            inputs.set_shape([None, self.input_width, len(self.input_columns)])
            labels.set_shape([None, self.label_width, len(self.label_columns)])

        return (inputs, known, date_inputs, date_labels), labels

    @property
    def train_ds(self) -> pd.DataFrame:
        return getattr(self, "_train_ds", pd.DataFrame())

    @property
    def valid_ds(self) -> pd.DataFrame:
        return getattr(self, "_valid_ds", pd.DataFrame())

    @property
    def test_ds(self) -> pd.DataFrame:
        return getattr(self, "_test_ds", pd.DataFrame())

    @property
    def train(self) -> tf.data.Dataset:
        """Build the train dataset."""
        return self._make_dataset(self.train_ds, self.batch_size)

    @property
    def valid(self) -> tf.data.Dataset:
        """Build the valid dataset."""
        if self.valid_ds.empty:
            raise AttributeError(
                "The validation dataset is empty. Try a positive value for `valid_size`."
            )
        return self._make_dataset(self.valid_ds, self.batch_size)

    @property
    def test(self) -> tf.data.Dataset:
        """Build the test dataset."""
        if self.test_ds.empty:
            raise AttributeError(
                "The test dataset is empty. Try a positive value for `test_size`."
            )
        return self._make_dataset(self.test_ds, self.batch_size)

    def forecast(
        self, data: pd.DataFrame, batch_size: Union[int, None]
    ) -> tf.data.Dataset:
        """
        Build the production dataset.

        Parameters
        ----------
        data : `dataframe of shape (steps, variables)`
            Data to forecast. It raises an error
            if not all columns defined by the instanciating are inside data.

        Returns
        -------
        data : `dataset`
            MapDataset which returns data with shape
            ((inputs, known, date_inputs, date_labels), labels).
        """

        assert all(
            self._filtered_data.columns.isin(data.columns)
        ), f"Data columns doesn't match the expected columns, got {data.columns}. Expected at least {self.filtered_data.columns}"
        data = data.loc[:, self.column_indices]
        data = self._make_dataset(data, batch_size)
        return data

    @property
    def data(self):
        """Return the original data"""
        return self._data

    @property
    def filtered_data(self):
        """Return the data filtered by columns"""
        return self._filtered_data

    def __validate__(
        self, method_name, args, kwargs
    ):  # pylint: disable=unused-argument
        """Validates attributes and args."""
        getattr(self, '_val' + method_name, lambda args, kwargs: True)(args, kwargs)
    
    def _val_compute_window_parameters(self, args, kwargs): # pylint: disable=unused-argument
        """Validates attributes and args of _compute_window_parameters method."""
        assert all(
            [col in self.data.columns for col in self.input_columns]
            + [True if len(self.input_columns) > 0 else False]
        ), f"The input columns are not found inside data, got {self.input_columns}, expected one or multiple choices of {self.data.columns}."
        assert all(
            [
                col in self.data.columns if len(self.label_columns) > 0 else False
                for col in self.label_columns
            ]
            + [True if len(self.label_columns) > 0 else False]
        ), f"""The label columns are not found inside data, got {self.label_columns}, expected one or multiple choices of {self.data.columns}."""
        assert all(
            [col in self.data.columns for col in self.known_columns]
        ), f"The known columns are not found inside data, got {self.known_columns}, expected one or multiple choices of {self.data.columns}."
        assert all(
            [col in self.data.columns for col in self.date_columns]
        ), f"The date columns are not found inside data, got {self.date_columns}, expected one or multiple choices of {self.data.columns}."
        assert (
            self.valid_start > 0
            and (self.valid_start - self.total_window_size + self.input_width)
            > 0  # self.input_width because we count the input width of the validation dataset
        ), f"""Not enough data for training dataset because valid dataset start at {self.valid_start}. Try to reduce valid size, test size or input width and label width."""

    def _val__init__(self, args, kwargs): # pylint: disable=unused-argument
        """Validates attributes and args of __init__ method."""
        assert (
            not self.data.empty
        ), f"The given parameter `data` is an empty DataFrame."
        assert (
            self.input_width > 0
        ), f"The input width has to be strictly positive, got {self.input_width}."
        assert (
            self.label_width > 0
        ), f"The label width has to be strictly positive, got {self.label_width}."
        assert (
            self.shift > 0
        ), f"The shift has to be strictly positive, got {self.shift}."
        assert (
            self.input_width <= self.data.shape[0] - self.shift
        ), f"The input width has to be equal or lower than {self.data.shape[0] - self.shift}, got {self.input_width}."
        assert (
            self.label_width < self.total_window_size
        ), f"The label width has to be equal or lower than {self.total_window_size}, got {self.label_width}"
        assert (
            self.test_size >= 0
        ), f"The test size has to be positive or null, got {self.test_size}."
        assert (
            self.valid_size >= 0
        ), f"The valid size has to be positive or null, got {self.valid_size}."
        assert (
            self.strategy in STRATEGIES
        ), f"Invalid strategy, got {self.strategy}, expected one of {STRATEGIES}."
        if self.batch_size:
            assert (
                self.batch_size > 0
            ), f"The batch size has to be strictly positive, got {self.batch_size}."

        assert not self.train_ds.empty, f"""The training dataset is empty, please redefine the test size or valid size.
                                        Got {self.test_size}, {self.valid_size} which lead
                                        to a test start: {self.test_start} and a valid start: {self.valid_size}."""

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

            Parameters remainder:\n
            - input_width : {self.input_width}
            - label_width : {self.label_width}
            - shift : {self.shift}
            - test_size : {self.test_size}
            - valid_size : {self.valid_size}
            - batch_size : {self.batch_size} \n

            The train set becomes : \n {self.train_ds} \n
            The validation set becomes : \n {self.valid_ds} \n
            The test set becomes : \n {self.test_ds} \n

            A split example: \n
                Inputs : \n {inputs}
                Known inputs : \n {known}
                Input dates : \n {date_inputs}
                Label dates : \n {date_labels}
                Labels : \n {labels}

                \n Label indices \n {self.label_indices}
                \n Input indices \n {self.input_indices}
                """
