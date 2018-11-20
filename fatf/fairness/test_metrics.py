# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:35:03 2018

@author: rp13102
"""

import pytest
import numpy as np

from metrics import get_confusion_matrix

input0 = [0, 0, 0, 1, 1, 1]
input1 = [1, 1, 0, 1, 0, 1]

labels0 = [1, 0]
output0 = [[2, 1],
           [2, 1]]

labels1 = [0, 1]
output1 = [[1, 2],
           [1, 2]]

@pytest.mark.parametrize("input0, input1, labels, expected_output",
                         [(input0, input1, labels0, output0),
                          (input0, input1, labels1, output1)])
def test_get_confusion_matrix(input0, input1, labels, expected_output):
    output_list = [item for sublist in expected_output for item in sublist]
    output_cm = get_confusion_matrix(input0, input1, labels)
    output_cm_list = [item for sublist in output_cm.tolist() for item in sublist]
    assert np.all([output_list[i] == output_cm_list[i] for i in range(len(output_list))])