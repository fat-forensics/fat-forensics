# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:23:40 2018

@author: rp13102
"""

import pytest
from fatf.interpretability.describe import describe_numeric, describe_categorical, describe_dataset
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
        
input_describe_categorical_0 = np.array(['a', 'b', 'a'])
expected_describe_categorical_0 = {'count' : 3,
                                'count_unique': 2,
                                'unique': ['a', 'b'],
                                'most_common': 'a',
                                'most_common_count': 2,
                                 'hist': {'a': 2, 'b':1},
                                 }


@pytest.mark.parametrize("input_series, expected_output",
                         [(input_describe_categorical_0, expected_describe_categorical_0)
                          ])
def test_describe_categorical(input_series, expected_output):
    output = describe_categorical(input_series)
    for key, val in expected_output.items():
        assert key in output.keys()
        assert np.all(val == output[key])
  
input_dataset0 = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), 
             ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), 
             ('diagnosis', '<U6'), ('dob', '<U10')])

expected_output0 = {'f': {'age': {'25%': 20.75,
                       '50%': 38.5,
                       '75%': 56.25,
                       'count': 2,
                       'max': 74,
                       'mean': 38.5,
                       'min': 3,
                       'std': 35.5}},
                     'm': {'age': {'25%': 34.5,
                       '50%': 39.0,
                       '75%': 43.5,
                       'count': 2,
                       'max': 48,
                       'mean': 39.0,
                       'min': 30,
                       'std': 9.0}}} 

condition0 = np.array(['f', 'f', 'm', 'm'])
todescribe0 = ['age']     

expected_output1 = {'age': {'25%': 23.25,
                          '50%': 39.0,
                          '75%': 54.5,
                          'count': 4,
                          'max': 74,
                          'mean': 38.75,
                          'min': 3,
                          'std': 25.897635027160298}}

@pytest.mark.parametrize("input_dataset, condition, todescribe, expected_output",
                         [(input_dataset0, condition0, todescribe0, expected_output0),
                          (input_dataset0, None, todescribe0, expected_output1)])
def test_describe_dataset(input_dataset, condition, todescribe, expected_output):
    output = describe_dataset(input_dataset, todescribe=todescribe, condition=condition)
    for key, val in expected_output.items():
        assert key in output.keys()
        if type(val) == dict:
            for key2, val2 in val.items():
                assert key2 in output[key].keys()
                assert np.all(val2 == output[key][key2])

condition1 = np.array(['f', 'm'])
@pytest.mark.parametrize("input_dataset, condition, todescribe",
                         [(input_dataset0, condition1, todescribe0, )])                
def test_describe_dataset2(input_dataset, condition, todescribe):
    with pytest.raises(ValueError):
        describe_dataset(input_dataset, todescribe=todescribe, condition=condition)
    