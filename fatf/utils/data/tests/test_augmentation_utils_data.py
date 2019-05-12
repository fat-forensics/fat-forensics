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

expected_output_0 = [np.array([[63, 57]]), np.array([[0.15201248595867722]])]

input_dataset_1 = testdata_struct[:2]
input_y = y[:2]

test_instance_1 = testdata_struct[1]
test_y_1 = y[1]

expected_output_1 = [np.array([(63, 57)], dtype=[('Age', '<i4'), ('Weight', '<i4')]), 
                     np.array([[0.15201248595867722]])]


@pytest.mark.parametrize("input_dataset, input_y, test_instance, test_y, expected_output",
                         [(input_dataset_0, input_y, test_instance_0, test_y_0, expected_output_0),
                          (input_dataset_1, input_y, test_instance_1, test_y_1, expected_output_1)])
def test_sample(input_dataset, input_y, test_instance, test_y, expected_output):
    mdl = Mixup(input_dataset,
                input_y)
    N = 1
    np.random.seed(42)
    output = mdl.sample(test_instance,
                        N)
    assert np.all(output[0] == expected_output[0])
    assert np.all(output[1] == expected_output[1])

input_t = np.vstack((input_y, 1-input_y))
test_t_0 = input_t[0]
test_instance_2cl_0 = input_dataset_0[0]
test_instance_2cl_1 = input_dataset_1[0]

expected_output_0 = (np.array([[27, 74]]), np.array([[0.6492304718478048, 0.35076952815219514]]))
expected_output_1 = (np.array([(27, 74)], dtype=[('Age', '<i4'), ('Weight', '<i4')]), np.array([[0.6492304718478048, 0.35076952815219514]]))
@pytest.mark.parametrize("input_dataset, input_t, test_instance, test_y, expected_output",
                         [(input_dataset_0, input_t, test_instance_2cl_0, test_t_0, expected_output_0),
                          (input_dataset_1, input_t, test_instance_2cl_1, test_t_0, expected_output_1)])
def test_sample_2classes(input_dataset, input_t, test_instance, test_y, expected_output):
    mdl = Mixup(input_dataset,
                input_t)
    np.random.seed(53)
    output = mdl.sample(test_instance,
                        test_y,
                        1)
    assert np.all(output[0] == expected_output[0])
    assert np.all(output[1] == expected_output[1])
  
input_t = np.vstack((input_y, 1-input_y))
test_t_0 = input_t[0]
input_dataset_cat = np.array([(74, 'a'), ( 3, 'b')],
      dtype=[('Age', '<i4'), ('work', '<U2')])

test_instance_cat_0 = input_dataset_cat[0]
expected_output_0 = (np.array([(20, 'a')], dtype=[('Age', '<i4'), ('work', '<U2')]), np.array([[0.74975274, 0.25024726]]))
@pytest.mark.parametrize("input_dataset, input_t, test_instance, test_y, expected_output",
                         [(input_dataset_cat, input_t, test_instance_cat_0, test_t_0, expected_output_0)])
def test_sample_categorical(input_dataset, input_t, test_instance, test_y, expected_output):
    mdl = Mixup(input_dataset,
                input_t)
    np.random.seed(7)
    output = mdl.sample(test_instance,
                        test_y,
                        1)
    assert np.all(output[0] == expected_output[0])
    assert float(format(output[1][0][0], '.8f')) == expected_output[1][0][0]
    assert float(format(output[1][0][1], '.8f')) == expected_output[1][0][1]
    
    
@pytest.mark.parametrize("input_dataset, input_y",
                         [(input_dataset_0, input_y)])   
def test_beta_parameters(input_dataset, input_y):
    with pytest.raises(ValueError) as excinfo:          
        mdl = Mixup(input_dataset,
                    input_y,
                    beta_parameters=[1])
    assert str(excinfo.value) == 'Need two parameters for beta distribution' 

    with pytest.raises(ValueError) as excinfo:          
        mdl = Mixup(input_dataset,
                    input_y,
                    beta_parameters=[-1, 1])
    assert str(excinfo.value) == 'Beta parameters need to be positive' 
    
    with pytest.raises(TypeError) as excinfo:          
        mdl = Mixup(input_dataset,
                    input_y,
                    beta_parameters=['a', 1])
    assert str(excinfo.value) == 'Beta parameters need to be int or float'