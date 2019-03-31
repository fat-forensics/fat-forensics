# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:23:40 2018

@author: rp13102
"""

import pytest

import numpy as np

import fatf.transparency.data.describe as ftdd

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
    output = ftdd.describe_numeric(input_series)
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
    output = ftdd.describe_categorical(input_series)
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
    output = ftdd.describe_dataset(input_dataset, todescribe=todescribe, condition=condition)
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
        ftdd.describe_dataset(input_dataset, todescribe=todescribe, condition=condition)

def test_generic():
    testdata = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '31/12/2015'),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 'male', '3131', 'heart', '02/10/2011'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),
       ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 'female', '0101', 'heart', '15/12/2015'),
       ('Monica Fry', 'morenocraig@howard.com', 24,  1, 'male', '1212', 'hip', '21/12/2005'),
       ('Michael Smith', 'edward72@dunlap-jackson.c', 44, 66, 'male', '0111', 'hip', '07/11/2012'),
       ('Dean Campbell', 'michele18@hotmail.com', 62, 96, 'female', '2320', 'lung', '22/01/2009'),
       ('Kimberly Kent', 'wilsoncarla@mitchell-gree', 63, 51, 'male', '2003', 'cancer', '16/06/2017'),
       ('Michael Burnett', 'collin04@scott.org', 26, 88, 'male', '0301', 'heart', '07/03/2009'),
       ('Patricia Richard', 'deniserodriguez@hotmail.c', 94, 64, 'female', '3310', 'heart', '20/08/2006'),
       ('Joshua Ramos', 'michaelolson@yahoo.com', 59, 19, 'female', '3013', 'cancer', '22/07/2005'),
       ('Samuel Fletcher', 'jessicagarcia@hotmail.com', 14, 88, 'female', '1211', 'lung', '29/07/2004'),
       ('Donald Hess', 'rking@gray-mueller.com', 16, 15, 'male', '0102', 'hip', '16/09/2010'),
       ('Rebecca Thomas', 'alex57@gmail.com', 94, 48, 'female', '0223', 'cancer', '05/02/2000'),
       ('Hannah Osborne', 'ericsullivan@austin.com', 41, 25, 'female', '0212', 'heart', '11/06/2012'),
       ('Sarah Nelson', 'davidcruz@hood-mathews.co', 36, 57, 'female', '0130', 'cancer', '13/01/2003'),
       ('Angela Kelly', 'pwilson@howell-bryant.com', 37, 52, 'female', '1023', 'heart', '28/03/2009'),
       ('Susan Williams', 'smithjoshua@allen.com', 21, 42, 'male', '0203', 'lung', '15/11/2005')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

    testdata2 = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'),
             ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'),
             ('diagnosis', '<U6'), ('dob', '<U10')])

    a=ftdd.describe_dataset(testdata2, todescribe=['age'], condition=np.array(['f', 'f', 'm', 'm']))
    print(a)

    a=ftdd.describe_dataset(testdata2, todescribe=['age'])
    print(a)
