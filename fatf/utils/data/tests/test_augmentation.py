# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:30:20 2018

@author: rp13102
"""

from fatf.utils.data.augmentation import Mixup
import pytest
import numpy as np

y = np.array([0, 1])
testdata_struct = np.array([(74, 52), ( 3, 86), (26, 56), (70, 57), (48, 57), (30, 98),
       (41, 73), (24,  1), (44, 66), (62, 96), (63, 51), (26, 88),
       (94, 64), (59, 19), (14, 88), (16, 15), (94, 48), (41, 25),
       (36, 57), (37, 52), (21, 42)],
      dtype=[('Age', '<i4'), ('Weight', '<i4')])

testdata = np.array([(74, 52), ( 3, 86), (26, 56), (70, 57), (48, 57), (30, 98),
       (41, 73), (24,  1), (44, 66), (62, 96), (63, 51), (26, 88),
       (94, 64), (59, 19), (14, 88), (16, 15), (94, 48), (41, 25),
       (36, 57), (37, 52), (21, 42)])

input_dataset_0 = testdata[:2]
input_y = y[:2]

test_instance_0 = testdata[1]
test_y_0 = y[1]

expected_output_0 = (np.array([[10, 10]]), 1)

@pytest.mark.parametrize("input_dataset, input_y, test_instance, test_y, expected_output",
                         [(input_dataset_0, input_y, test_instance_0, test_y_0, expected_output_0)])
def test_sample(input_dataset, input_y, test_instance, test_y, expected_output):
    mdl = Mixup(input_dataset,
                input_y)
    N = 1
    output = mdl.sample(test_instance,
                        N)
    assert np.all(output[0] == expected_output[0])
    assert output[1] == expected_output[1]
    