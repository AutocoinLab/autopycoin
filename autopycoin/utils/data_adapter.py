"""Functions which compute convenients data processing"""

import tensorflow as tf
        

def unpack_timeseries(data):
    """Unpacks user-provided data tuple.
       
    This is a convenience utility to be used when overriding
    `Model.test_step`, `Model.train_step` or `Model.predict_step`.
    This utility makes it easy to support inputs of the form `(inputs, known, date_inputs, date_labels)`
    or `(inputs, date_inputs, date_labels)`
    """    
    
    # Numpy format
    if len(data) == 3:
        return data[0]
    elif len(data) == 4:
        return (data[0], data[1])
    elif isinstance(data, tf.Tensor):
        return data
    else:
        error_msg = ("Data is expected to be in format `inputs, date_inputs, date_labels`"
                     "or `inputs, known, date_inputs, date_labels`, found: {}").format(data)
        raise ValueError(error_msg)