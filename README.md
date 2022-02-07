# Welcome to Autopycoin
---

![ci_autopycoin](https://github.com/AutocoinLab/autopycoin/actions/workflows/ci_autopycoin.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/AutocoinLab/autopycoin/tree/main.svg?style=svg)](https://circleci.com/gh/AutocoinLab/autopycoin/tree/main)
[![codecov](https://codecov.io/gh/AutocoinLab/autopycoin/branch/main/graph/badge.svg?token=5ZE3XddqtL)](https://codecov.io/gh/AutocoinLab/autopycoin)
[![Documentation Status](https://readthedocs.org/projects/autopycoin/badge/?version=latest)](https://autopycoin.readthedocs.io/en/latest/?badge=latest)


This is a deep learning package based on tensorflow and 
maintained by a group of french students and used in a their final master project.
All the models are the Unofficial implementations.

# Available Models
---

```python
from autopycoin.models import create_interpretable_nbeats, create_interpretable_nbeats, NBEATS, PoolNBEATS
```

| Model         | epistemic error | Aleotoric error | Paper |
|--------------|-----------|------------|------------|
| NBEATS |  Dropout or bagging   | Quantiles |  [Paper](https://arxiv.org/abs/1905.10437)|


# Available Losses
---

```python
from autopycoin.losses import QuantileLossError, SymetricMeanAbsolutePercentageError
```

| Losses         |
|--------------|
| QuantileLosseError | 
| SymetricMeanAbsolutePercentageError | 

# Dataset maker
---

```python
from autopycoin.dataset import WindowGenerator
```

| Dataset maker|
|--------------|
| WindowGenerator|

# How to use autopycoin
---

```
pip install autopycoin
```

## Univariate time series
---

```python
from autopycoin.models import create_interpretable_nbeats
from autopycoin.losses import QuantileLossError
from autopycoin.dataset import WindowGenerator

import pandas as pd

# Let's suppose we have a pandas time series
data = pd.DataFrame([
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
    [7, 7],
    [8, 8],
    [9, 9],
    [10, 10]
    ])

w_oneshot = WindowGenerator(input_width=3, # Input width used for the model
                            label_width=2, # label width used for the model
                            shift=2, # if shift < label_width then input and label will overlap each other
                            valid_size=2, # Defines your validation set size (if int then this is the number of instances else float this the percentage of your dataset)
                            test_size=1, # The same
                            flat=True, # flat to true to flatten the input (don't use multivariate time series) 
                            batch_size=None,
                            preprocessing=None # function to apply to the input  
                            )

# Let suppose that tthe columns 0 is the one we want to predict
w_oneshot = w_oneshot.from_array(
    data,
    input_columns=[0],
    label_columns=[0])

# w_oneshot is then our dataset maker

model = create_interpretable_nbeats(label_width=w_oneshot.label_width)

model.compile(loss=QuantileLossError(quantiles=[0.5]))

# fit
model.fit(w_oneshot.train, validation_data=w_oneshot.valid)

model.predict(w_oneshot.test)
```

## Multivariate time series
---

```python
from autopycoin.models import create_interpretable_nbeats
from autopycoin.losses import QuantileLossError
from autopycoin.dataset import WindowGenerator

import pandas as pd

# The same example
data = pd.DataFrame([
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
    [7, 7],
    [8, 8],
    [9, 9],
    [10, 10]
    ])

w_oneshot = WindowGenerator(input_width=3, # Input width used for the model
                            label_width=2, # label width used for the model
                            shift=2, # if shift < label_width then input and label will overlap each other
                            valid_size=2, # Defines your validation set size (if int then this is the number of instances else float this the percentage of your dataset)
                            test_size=1, # The same
                            flat=False, # flat to False
                            batch_size=None,
                            preprocessing=None # function to apply to the input  
                            )

# This is an univariate model hence input_columns and label_columns has to be equal
w_oneshot = w_oneshot.from_array(
    data,
    input_columns=[0, 1], 
    label_columns=[0, 1])

# w_oneshot is then our dataset maker

model = create_interpretable_nbeats(label_width=w_oneshot.label_width)

model.compile(loss=QuantileLossError(quantiles=[0.5]))

# fit
model.fit(w_oneshot.train, validation_data=w_oneshot.valid)

model.predict(w_oneshot.test) # The last dim corresponds to the variables
```