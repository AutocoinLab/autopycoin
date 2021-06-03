import tensorflow as tf

class WindowGenerator():
    def __init__(self, 
                 input_width,
                 label_width, 
                 shift, 
                 df, 
                 valid_size, 
                 test_size, 
                 batch_shape,
                 input_columns=None,
                 known_columns=None,
                 label_columns=None,
                 date_columns=None):

        # Work out the label column indices.
        self.input_columns = input_columns
        self.known_columns = known_columns
        self.label_columns = label_columns
        self.date_columns = date_columns
        self.columns = set(input_columns+known_columns+label_columns+date_columns)
        
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            
        self.column_indices = {name: i for i, name in
                               enumerate(self.columns)}

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
        self.test_size= test_size
        
        self.train_df = df.iloc[:-(test_size+val_size+label_width)].loc[:, self.columns]
        self.valid_df = df.iloc[-(self.total_window_size+val_size+label_width):-(test_size+val_size+label_width-1)].loc[:, self.columns]
        self.test_df = df.iloc[-(self.total_window_size+test_size):].loc[:, self.columns]
        
        self.batch_shape = batch_shape
        
        print(self)
    
    def split_window(self, features):
        
        inputs = features[:, self.input_slice, :] # Batch_size, row, col
        labels = features[:, self.labels_slice, :]
        
        date = tf.stack([features[:, :, self.column_indices[name]] for name in self.date_columns], axis=-1)
        
        inputs = tf.stack([inputs[:, :, self.column_indices[name]] for name in self.input_columns], axis=-1)
        known = tf.stack([labels[:, :, self.column_indices[name]] for name in self.known_columns], axis=-1)
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        known.set_shape([None, self.label_width, None])
        labels.set_shape([None, self.label_width, None])
        date.set_shape([None, self.total_window_size, None])
        date = tf.strings.as_string(date)
        date = tf.strings.reduce_join(date, separator='-', axis=2)
        date_inputs = date[:, self.input_slice]
        date_labels = date[:, self.labels_slice]

        return (inputs, known, date_inputs, date_labels), labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.int64)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,
                                                                  targets=None,
                                                                  sequence_length=self.total_window_size,
                                                                  sequence_stride=1,
                                                                  shuffle=False,
                                                                  batch_size=self.batch_shape)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def valid(self):
        return self.make_dataset(self.valid_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)

    def __repr__(self):
        
        for (inputs, known, date_inputs, date_labels), labels in self.train.take(1):
            return f"""Starting window generator\n

            Inputs are : {self.input_columns}
            Known inputs are : {self.known_columns}
            Outputs are : {self.label_columns}
            Dates are : {self.date_columns} \n

            Associated indices to each column: {self.column_indices} \n

            Parameters remainder:\n
            - input_width : {self.input_width}
            - label_width : {self.label_width} 
            - shift : {self.shift}
            - test_size : {self.test_size} 
            - valid_size : {self.valid_size}
            - batch_shape : {self.batch_shape} \n

            Training set become : \n {self.train_df} \n
            Validation set become : \n {self.valid_df} \n
            Test set become : \n {self.test_df} \n

            Split example: \n
                Inputs : \n {inputs.numpy(), inputs.shape}
                Known inputs : \n {known.numpy(), known.shape}
                Inputs dates : \n {date_inputs.numpy(), date_inputs.shape}
                Outputs dates : \n {date_labels.numpy(), date_labels.shape}
                Outputs : \n {labels.numpy(), labels.shape}"""
        

    def plot(self, window_generator, plot_col, model=None, max_subplots=3):

        for (inputs, known, date_inputs, date_labels), labels in window_generator.take(1):
            
            date_inputs, date_labels = date_inputs.numpy(), date_labels.numpy()
            fig = plt.figure(figsize=(12, 8))
            
            if self.label_columns:
                plot_col_index = self.label_columns_indices.get(plot_col, None)
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                plot_col_index = self.column_indices[plot_col]
                label_col_index = plot_col_index
            
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(max_n, 1, n+1)
                plt.ylabel(f'{plot_col}')
                plt.plot(date_inputs[n], inputs[n, :, plot_col_index],
                         label='Inputs', marker='.', zorder=-10)

                if label_col_index is None:
                    continue

                plt.scatter(date_labels[n], labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model.predict(window_generator)
                    plt.scatter(date_labels[n], predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

                if n == 0:
                    plt.legend()
                plt.xticks(rotation='vertical')

            fig.tight_layout()
            plt.xlabel('Time [h]')
        
    def table(self, window_generator, plot_col, model=None, num_table=3):
        
        for (inputs, known, date_inputs, date_labels), labels in window_generator.take(1):
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
                
            results = pd.DataFrame(index=date_labels[num_table], columns=['True', 'Pred', 'mape'])
            results.loc[:, 'True'] = labels[num_table].numpy()
            results.loc[:, 'Pred'] = model.predict(window_generator)[num_table, :, label_col_index]
            results.loc[:, 'mape'] = abs(results.loc[:, 'True'] - results.loc[:, 'Pred']) / results.loc[:, 'True']

        return results

