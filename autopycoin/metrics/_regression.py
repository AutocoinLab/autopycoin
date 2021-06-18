# -*- coding: utf-8 -*-
"""Metrics to assess performance on regression task.
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

import tensorflow as tf


__ALL__ = [
    "smape",
    "mase",
    "owa"
]

