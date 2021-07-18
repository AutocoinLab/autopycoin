import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ..keras import AutoRegressive

class WindowGenerator():
    """Transform a time series dataset into an usable format.
        
    Args:
      data: Dataframe of shape (timesteps, variables).
        The time series dataframe.
      input_width: Integer.
        The number of historical time steps to use in the model.
      label_width: Integer
        the number of time steps to forecast.
      shift: Integer
        Compute the shift between inputs variables and labels variables.
      valid_size: Integer.
        The Number of examples in the validation set.
      test_size: Integer.
        The Number of examples in the test set.
      batch_size: Integer.
        The number of examples per batch.
      input_columns: List of str.
        The input columns names, default to None.
      known_columns: List of str.
        The known columns names, default to None.
      label_columns: List of str.
        The label columns names, default to None.
      date_columns: List of str.
        The date columns names, default to None. 
        Date columns will be cast to string and join by '-' delimiter to be used as xticks.
        
    """
    
    def __init__(self, data,
                       input_width,
                       label_width, 
                       shift, 
                       test_size, 
                       valid_size, 
                       batch_size=None,
                       input_columns=None,
                       known_columns=None,
                       label_columns=None,
                       date_columns=None):
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        self.valid_size = valid_size
        self.test_size = test_size 
        
        self.batch_size = batch_size
        
        # -1 pour se décaler de 1 étape temporelle par rapport à la validation
        self.train_ds = data.iloc[:-(test_size+valid_size-1+label_width)] 
        # si test_size=1 donc seulement label_width qui correspond au test
        self.valid_ds = data.iloc[-(self.total_window_size+valid_size+label_width+test_size-2):-(label_width+test_size-1)]
        # si test_size=1 donc seulement total_window_size qui contient 1 exemple
        self.test_ds = data.iloc[-(self.total_window_size+test_size-1):]

        # Work out the column indices.
        self.known_columns = known_columns
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.date_columns = date_columns
        
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)} # Column indices according to the label dataset
        
        self.inputs_columns_indices = {name: i for i, name in enumerate(input_columns)} # Column indices according to the input dataset
           
        # label indices according to the input dataset 
        self.labels_in_inputs_indices = {key: value for key, value in self.inputs_columns_indices.items() if key in label_columns} 

        self.column_indices = {name: i for i, name in enumerate(self.train_ds)} # Columns indices according to the dataset
    
    def make_dataset(self, data):
        
        """Compute the tensorflow dataset object.
        
        Args:
          data: dataframe or array or tensor of shape (timestep, variables)
            The time series dataset.
            
        Returns:
          ds: Tensorflow dataset.
            The dataset that can be used in keras model.
        """
        
        if self.batch_size is None:
            batch_size = len(data)
        
        else:
            batch_size = self.batch_size
            
        # Necessary because ML model need all values   
        data = np.array(data, dtype=np.float64)
        
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,
                                                                  targets=None,
                                                                  sequence_length=self.total_window_size,
                                                                  sequence_stride=1,
                                                                  shuffle=False,
                                                                  batch_size=batch_size)

        ds = ds.map(self.split_window)

        return ds
    
    def split_window(self, features):
        
        """Compute the window split.
        
        Args:
         features: Tensor of shape(Batch_size, timestep, variables)
            The window define by timeseries_dataset_from_array class.
            
        Returns:
          inputs: Tensor of shape (batch_size, input_width, variables)
            The inputs variables. 
          known: Tensor of shape (batch_size, label_width, variables)
            The known variables.
          date_inputs: Tensor of shape (batch_size, input_width, 1)
            Date of the inputs, default to range of shape (input_width, 1).
          date_labels: Tensor of shape (batch_size, label_width, 1)
            Date of the labels, default to range of shape (label_width, 1).
          labels: Tensor of shape (batch_size, label_width, variables)
            The Outputs variables.
        
        """

        # Workout Date 
        if self.date_columns is not None:
            date = tf.stack([features[:, :, self.column_indices[name]] for name in self.date_columns], axis=-1)
        else: 
            date = tf.range(features.shape[0]).reshape(-1, 1)
            
        date.set_shape([None, self.total_window_size, None])
        date = tf.strings.as_string(date)
        date = tf.strings.reduce_join(date, separator='-', axis=2)
        date_inputs = date[:, self.input_slice]
        date_labels = date[:, self.labels_slice]
        
        # Workout Known inputs
        if self.known_columns is not None:
            known = tf.stack([features[:, self.labels_slice, self.column_indices[name]] for name in self.known_columns], axis=-1)
            known.set_shape([None, self.label_width, None])
        else:
            known = None
           
        # Workout inputs and labels
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        inputs = tf.stack([inputs[:, :, self.column_indices[name]] for name in self.input_columns], axis=-1)
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return (inputs, known, date_inputs, date_labels), labels
    
    @property
    def train(self):
        """ Compute the train dataset.
        """
        return self.make_dataset(self.train_ds)
    
    @property
    def valid(self):
        """ Compute the valid dataset.
        """
        return self.make_dataset(self.valid_ds)

    @property
    def test(self):
        """ Compute the test dataset.
        """
        return self.make_dataset(self.test_ds)
    
    def forecast(self, data):
        """ Compute the production dataset.
        """
        return self.make_dataset(data)
    
    def plot(self,
             dataset, 
             plot_col,
             plot_labels=True,
             plot_history=None,
             model=None,
             max_subplots=3):

        """ Display the results with matplotlib.
        """

        for inputs, labels in dataset.take(1):

            # We need to preprocess inputs and labels to create inputs with shape (batch, timesteps, variables)
            # Which is not the case for OneShot model because input is in the right format.

            if isinstance(model, AutoRegressive):
                inputs, labels = model.numpy_format(inputs, labels)
                inputs, labels = model.predict_format(inputs, labels)
                    
                (inputs, date_inputs, date_labels) = inputs
              
            # Format DL
            else:
                (inputs, known, date_inputs, date_labels) = inputs
            
            date_inputs, date_labels = date_inputs.numpy(), date_labels.numpy()
            fig = plt.figure(figsize=(12, 8))

            # Get label and plot column indices 
            if self.label_columns:
                plot_col_index = self.inputs_columns_indices.get(plot_col, None)
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                plot_col_index = self.column_indices[plot_col]
                label_col_index = plot_col_index

            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(max_n, 1, n+1)
                plt.ylabel(f'{plot_col}')

                if plot_history is not None:
                    plt.plot(plot_history[0], plot_history[1],
                             label='Entrées', marker='.', zorder=-10)
                else: 
                    print(inputs)
                    plt.plot(date_inputs[n, :self.input_width], inputs[n, :self.input_width, plot_col_index],
                             label='Entrées', marker='.', zorder=-10)

                if label_col_index is None:
                    continue

                if plot_labels is True:
                    plt.plot(date_labels[n], labels[n, ..., label_col_index],
                                c='#2ca02c')

                    plt.scatter(date_labels[n], labels[n, ..., label_col_index],
                                edgecolors='k', label='valeurs réelles', c='#2ca02c', s=64)

                if model is not None:
                    predictions = model.predict(dataset)

                    plt.plot(date_labels[n], predictions[n, ..., label_col_index],
                              c='#ff7f0e')

                    plt.scatter(date_labels[n], predictions[n, ..., label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

                if n == 0:
                    plt.legend()
                plt.xticks(rotation='vertical')

            fig.tight_layout()
            plt.xlabel('Time')

    def table(self, 
              dataset,
              plot_col,
              model=None,
              num_table=3):

        """ Display the results as a dataframe.
        """

        for (inputs, known, date_inputs, date_labels), labels in dataset.take(1):

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            results = pd.DataFrame(index=date_labels[num_table-1], columns=['True', 'Pred', 'mape'])
            results.loc[:, 'True'] = labels[num_table-1].numpy()
            results.loc[:, 'Pred'] = model.predict(dataset)[num_table-1, :, label_col_index]
            results.loc[:, 'mape'] = abs(results.loc[:, 'True'] - results.loc[:, 'Pred']) / results.loc[:, 'True']

        return results
    
    def __repr__(self):
        """ Display some explanations.
        """
            
        iterator = iter(self.train)
        ((inputs, known, date_inputs, date_labels), labels) = iterator.get_next()
        
        return f"""Début du générateur de fenêtre \n

            Les colonnes d'entrées sont : {self.input_columns}
            Les colonnes connues sont : {self.known_columns}
            Les colonnes sorties sont : {self.label_columns}
            Les colonnes dates sont : {self.date_columns} \n

            Les indices associés à chaque colonne sont: {self.column_indices} \n

            Rappel des paramètres:\n
            - input_width : {self.input_width}
            - label_width : {self.label_width} 
            - shift : {self.shift}
            - test_size : {self.test_size} 
            - valid_size : {self.valid_size}
            - batch_size : {self.batch_size} \n

            Le set d'entrainement devient : \n {self.train_ds} \n
            Le set de validation devient : \n {self.valid_ds} \n
            Le set de test devient : \n {self.test_ds} \n

            Exemple de split avec leur indices: \n
                entrées : \n {inputs}
                entrées connues : \n {known}
                dates entrées : \n {date_inputs}
                date sorties : \n {date_labels}
                sortie : \n {labels}
                
                \n indices des sorties \n {self.label_indices}
                \n indices des entrées \n {self.input_indices}"""
    
    
