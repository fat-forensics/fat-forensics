# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:47:33 2018

@author: rp13102
"""
import pytest
import numpy as np
from fatf.interpretability.cf_explainer import Explainer, combine_arrays

input_array0_0 = np.array([1,2,3])
input_array1_0 = np.array([4,5,6,7,8])
expected_output_0 = [1,4,2,5,3,6,7,8]

@pytest.mark.parametrize("array0, array1, expected_output",
                         [(input_array0_0, input_array1_0, expected_output_0)])
def test_combine_arrays(array0, array1, expected_output):
    output_array = combine_arrays(array0, array1)
    assert np.all(output_array == expected_output)