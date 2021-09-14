.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: Time series forecasting
==================================================

Our models
----------

This package aims to define multiple deep learning model for time series forecasting
using tensorflow. The first model created is :class:`autopycoin.models.N_BEATS`.
We provide really easy-to-use functions which help you to build this model.
It can be imported as::

    >>> from autopycoin.models import create_interpretable_nbeats,
                                      create_generic_nbeats


Our losses
----------

We defined some loss functions, currently available:

* Quantile loss function
* SymetricMeanAbsolutePercentageError
