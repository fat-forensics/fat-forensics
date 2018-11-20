# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:35:03 2018

@author: rp13102
"""

import pytest
import numpy as np

from metrics import get_confusion_matrix, filter_dataset, split_dataset

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
    
x0_input = np.array([(0, 0, 1), 
                 (1, 1, 0), 
                 (0, 1, 1), 
                 (0, 1, 0),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

x1_input = np.array([(0, 0, 1), 
                 (1, 1, 1), 
                 (0, 1, 1), 
                 (0, 1, 1),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

x2_input = np.array([(0, 0, 0), 
                 (1, 1, 0), 
                 (0, 1, 0), 
                 (0, 1, 0),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

targets = np.array([1, 1, 0, 0])
predictions = np.array([0, 1, 0, 1])

input_data0 = [x0_input, targets, predictions]
input_data1 = [x1_input, targets, predictions]
input_data2 = [x2_input, targets, predictions]

x0_output = np.array([(0, 0, 1), 
                      (0, 1, 1)],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

targets0_output = np.array([1, 0])
predictions0_output = np.array([0, 0])

x2_output = np.array([],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

expected_data0 = [x0_output, targets0_output, predictions0_output]
expected_data1 = [x1_input, targets, predictions]
expected_data2 = [x2_output, np.array([], dtype='int32'), np.array([], dtype='int32')]

@pytest.mark.parametrize("input_data, feature, feature_value, expected_data",
                         [(input_data0, 'gender', 1, expected_data0),
                          (input_data1, 'gender', 1, expected_data1),
                          (input_data2, 'gender', 1, expected_data2)])
def test_filter_dataset(input_data, feature, feature_value, expected_data):
    X, targets, predictions = input_data
    x_expected, targets_expected, predictions_expected = expected_data
    
    x_output, targets_output, predictions_output = \
        filter_dataset(X, targets, predictions, feature, feature_value)
    assert np.all(x_expected == x_output)
    assert np.all(targets_expected == targets_output)
    assert np.all(predictions_expected == predictions_output)
 
expected_splits0 = [(0, 
                     (np.array([(1, 1, 0), 
                                (0, 1, 0)],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([1, 0]),
                    np.array([1, 1]))
                      ),
                    (1, 
                     (x0_output,
                    np.array([1, 0]),
                    np.array([0, 0])
                      )
                     )
                    ]
    
expected_splits1 = [(0, 
                     (np.array([],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([]),
                    np.array([]))
                      ),
                    (1, 
                     (x1_input,
                      targets,
                      predictions
                      )
                     )
                    ]  
                     
expected_splits2 = [(0, 
                     (x2_input,
                      targets,
                      predictions
                      )
                     ),
                    (1, 
                     (np.array([],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([]),
                    np.array([]))
                      )
                    ]  
@pytest.mark.parametrize("input_data, feature, expected_splits",
                         [(input_data0, 'gender', expected_splits0),
                          (input_data1, 'gender', expected_splits1),
                          (input_data2, 'gender', expected_splits2)])
def test_split_dataset(input_data, feature, expected_splits):
    X, targets, predictions = input_data
    labels = [0, 1]
    output_splits = split_dataset(X, targets, predictions, feature, labels)
    for idx in range(2):
        x_expected, targets_expected, predictions_expected = expected_splits[idx][1]
        x_output, targets_output, predictions_output = output_splits[idx][1]
        
        assert expected_splits[idx][0] == output_splits[idx][0]
        assert np.all(x_expected == x_output)
        assert np.all(targets_expected == targets_output)
        assert np.all(predictions_expected == predictions_output)