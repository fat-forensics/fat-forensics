# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:23:40 2018

@author: rp13102
"""

import pytest
from describe import describe_numeric
import numpy as np
input_describe_numeric_0 = np.array([0], dtype='int32')
expected_describe_numeric_0 = {'count' : 1,
                                'mean': 0.0,
                                'std': 0.0,
                                'max': 0,
                                'min': 0,
                                 '25%': 0.0,
                                 '50%': 0.0,
                                 '75%': 0.0}

@pytest.mark.parametrize("input_series, expected_output",
                         [(input_describe_numeric_0, expected_describe_numeric_0)])
def test_describe_numeric(input_series, expected_output):
    output = describe_numeric(input_series)
    for key, val in expected_output.items():
        assert key in output.keys()
        assert val == output[key]