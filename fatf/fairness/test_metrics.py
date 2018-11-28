# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:35:03 2018

@author: rp13102
"""

import pytest
import numpy as np

from metrics import get_confusion_matrix, filter_dataset, split_dataset, get_cross_product
from metrics import get_mask, get_weights_costsensitivelearning, get_counts, apply_combination_filter
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

input_data_crossproduct = np.array([('a', 'A', '0'),
                        ('b', 'B', '1')],
                        dtype=[('first', '<U1'), 
                               ('second', '<U1'), 
                               ('third', '<U1')]
                        )  
expected_crossproduct0 = [('a',), ('b',)]
expected_crossproduct1 = [('a', 'A',), ('a', 'B',), ('b', 'A',), ('b', 'B',)]
@pytest.mark.parametrize("input_data, features_to_check, expected_output",
                         [(input_data_crossproduct, ['first'], expected_crossproduct0),
                          (input_data_crossproduct, ['first', 'second'], expected_crossproduct1)])   
def test_get_cross_product(input_data, features_to_check, expected_output):
    output = get_cross_product(input_data, features_to_check)
    assert set(output) == set(expected_output)

input_data_mask0 = np.array([('a', 'A', 0),
                               ('b', 'B', 1),
                               ('a', 'B', 0),
                               ('b', 'B', 1),
                               ('a', 'A', 0),
                               ('b', 'B', 1)],
                            dtype=[('first', '<U1'), 
                                   ('second', '<U1'), 
                                   ('third', '<i4')]
                            )  
expected_output_mask0 = [True, False, True, False, True, False]
expected_output_mask1 = [True, False, False, False, True, False]

@pytest.mark.parametrize("input_dataset, features_to_check, combination, expected_output",
                         [(input_data_mask0, ['first'], ('a',), expected_output_mask0),
                          (input_data_mask0, ['first', 'second'], ('a', 'A', ), expected_output_mask1)])
def test_get_mask(input_dataset, features_to_check, combination, expected_output):
    output = get_mask(input_dataset, features_to_check, combination)
    assert set(output) == set(expected_output)
 
expected_weights0 = np.array([1, 1, 1, 1, 1, 1], dtype=float).reshape(-1, 1)
expected_weights1 = np.array([2, 1, 1, 1, 2, 1], dtype=float).reshape(-1, 1)
expected_weights2 = np.array([1.5, 1, 3, 1, 1.5, 1], dtype=float).reshape(-1, 1)

@pytest.mark.parametrize("input_dataset, features_to_check, expected_weights", 
                         [(input_data_mask0, ['first'], expected_weights0),
                          (input_data_mask0, ['second'], expected_weights1),
                          (input_data_mask0, ['first', 'second'], expected_weights2)])
def test_get_weights_costsensitivelearning(input_dataset, features_to_check, expected_weights):
    target_field = 'third'
    cross_product = get_cross_product(input_dataset, features_to_check)
    counts = get_counts(input_dataset, target_field, features_to_check, cross_product)
    output_weights = get_weights_costsensitivelearning(input_dataset, features_to_check, counts)
    assert np.all(output_weights == expected_weights)

input_data_counts = np.array([('a', 'A', 0),
                               ('b', 'B', 1),
                               ('a', 'B', 0),
                               ('b', 'B', 1),
                               ('a', 'A', 0),
                               ('b', 'B', 0)],
                            dtype=[('first', '<U1'), 
                                   ('second', '<U1'), 
                                   ('third', '<i4')]
                            ) 
 
expected_counts0 = {('a',): {0: 3},
                    ('b',): {0: 1, 1: 2}}

expected_counts1 = {('a', 'A', ): {0: 2},
                    ('a', 'B', ): {0: 1},
                    ('b', 'B', ): {0: 1, 1: 2}                    
                    }

@pytest.mark.parametrize("input_dataset, features_to_check, expected_counts",
                         [(input_data_counts, ['first'], expected_counts0),
                          (input_data_counts, ['first', 'second'], expected_counts1)])
def test_get_counts(input_dataset, features_to_check, expected_counts):
    target_field = 'third'
    cross_product = get_cross_product(input_dataset, features_to_check)
    output_counts = get_counts(input_dataset, target_field, features_to_check, cross_product)
    for key, val in output_counts.items():
        assert dict(output_counts[key]) == expected_counts[key]

input_data_apply = np.array([('a', 'A', 0, 1),
                               ('b', 'B', 1, 1),
                               ('a', 'B', 0, 0),
                               ('b', 'B', 1, 0),
                               ('a', 'A', 0, 0),
                               ('b', 'B', 0, 1)],
                            dtype=[('first', '<U1'), 
                                   ('second', '<U1'), 
                                   ('third', '<i4'),
                                   ('fourth', '<i4')]
                            ) 

expected_apply0 = ([1, 0, 0],
                   [0, 0, 0])

expected_apply1 = ([1, 0],
                   [0, 0])
@pytest.mark.parametrize("input_dataset, features_to_check, combination, expected_output",
                         [(input_data_apply, ['first'], ('a', ), expected_apply0),
                          (input_data_apply, ['first', 'second'], ('a', 'A', ), expected_apply1)])        
def test_apply_combination_filter(input_dataset, features_to_check, combination, expected_output):
    target_field = 'third'
    prediction_field = 'fourth'
    output = apply_combination_filter(input_dataset, prediction_field, target_field, features_to_check, combination)
    assert np.all(output[0] == expected_output[0])
    assert np.all(output[1] == expected_output[1])