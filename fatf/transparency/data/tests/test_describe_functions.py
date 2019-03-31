"""
Tests describing arrays.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.transparency.data.describe_functions as ftddf

from fatf.exceptions import IncorrectShapeError

NUMERICAL_KEYS = ['count', 'mean', 'std', 'max', 'min', '25%', '50%', '75%',
                  'nan_count']
CATEGORICAL_KEYS = ['count', 'count_unique', 'unique', 'most_common',
                    'most_common_count', 'hist']


def test_describe_numerical_array():
    """
    Tests :func:`fatf.transparency.data.describe.describe_numerical_array`.
    """
    runtime_warning = 'Invalid value encountered in percentile'
    #
    incorrect_shape_error = 'The input array should be 1-dimensional.'
    value_error_non_numerical = 'The input array should be purely numerical.'
    value_error_empty = 'The input array cannot be empty.'

    # Wrong shape
    array = np.array([[5, 33], [22, 17]])
    with pytest.raises(IncorrectShapeError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == incorrect_shape_error

    # Wrong type
    array = np.array(['string', 33, 22, 17])
    with pytest.raises(ValueError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == value_error_non_numerical

    # Empty array
    array = np.array([], dtype=np.int32)
    with pytest.raises(ValueError) as exin:
        ftddf.describe_numerical_array(array)
    assert str(exin.value) == value_error_empty

    # Array with nans -- structured row; ignore nans + default parameter
    array = np.array([(33, 22, np.nan, 11, np.nan, 4)],
                     dtype=[('a', int), ('b', int), ('c', np.float),
                            ('d', np.int32), ('e', np.float), ('f', int)])
    description = {'count': 6, 'mean': 17.5, 'std': 11.011, 'max': 33,
                   'min': 4, '25%': 9.25, '50%': 16.5, '75%': 24.75,
                   'nan_count': 2}
    array_description = ftddf.describe_numerical_array(array[0])
    assert set(NUMERICAL_KEYS) == set(description.keys())
    assert set(NUMERICAL_KEYS) == set(array_description.keys())
    for i in NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]
    # ...
    array_description = ftddf.describe_numerical_array(
        array[0], skip_nans=True)
    assert set(NUMERICAL_KEYS) == set(description.keys())
    assert set(NUMERICAL_KEYS) == set(array_description.keys())
    for i in NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]


    # Array with nans -- classic array; do not ignore nans
    array = np.array([33, 22, np.nan, 11, np.nan, 4])
    description = {'count': 6, 'mean': np.nan, 'std': np.nan, 'max': np.nan,
                   'min': np.nan, '25%': np.nan, '50%': np.nan, '75%': np.nan,
                   'nan_count': 2}
    with pytest.warns(RuntimeWarning) as w:
        array_description = ftddf.describe_numerical_array(
            array, skip_nans=False)
    assert set(NUMERICAL_KEYS) == set(description.keys())
    assert set(NUMERICAL_KEYS) == set(array_description.keys())
    assert len(w) == 3
    for i in range(len(w)):
        assert str(w[i].message).startswith(runtime_warning)
    for i in NUMERICAL_KEYS:
        true = description[i]
        computed = array_description[i]
        if np.isnan(true) and np.isnan(computed):
            assert True
        else:
            assert true == computed

    # Array without nans -- classic array; ignore nans
    array = np.array([33, 22, 11, 4])
    description = {'count': 4, 'mean': 17.5, 'std': 11.011, 'max': 33,
                   'min': 4, '25%': 9.25, '50%': 16.5, '75%': 24.75,
                   'nan_count': 0}
    array_description = ftddf.describe_numerical_array(array, skip_nans=True)
    assert set(NUMERICAL_KEYS) == set(description.keys())
    assert set(NUMERICAL_KEYS) == set(array_description.keys())
    for i in NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]

    # Array without nans -- classic array; do not ignore nans
    array_description = ftddf.describe_numerical_array(array, skip_nans=False)
    assert set(NUMERICAL_KEYS) == set(description.keys())
    assert set(NUMERICAL_KEYS) == set(array_description.keys())
    for i in NUMERICAL_KEYS:
        assert pytest.approx(array_description[i], abs=1e-3) == description[i]


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
def test_describe_categorical_array(input_series, expected_output):
    """
    Tests :func:`fatf.transparency.data.describe.describe_categorical_array`.
    """
    return
    output = ftddf.describe_categorical_array(input_series)
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
def test_describe_array(input_dataset, condition, todescribe, expected_output):
    """
    Tests :func:`fatf.transparency.data.describe.describe_array`.
    """
    return
    output = ftddf.describe_array(input_dataset, todescribe=todescribe, condition=condition)
    for key, val in expected_output.items():
        assert key in output.keys()
        if type(val) == dict:
            for key2, val2 in val.items():
                assert key2 in output[key].keys()
                assert np.all(val2 == output[key][key2])

condition1 = np.array(['f', 'm'])
@pytest.mark.parametrize("input_dataset, condition, todescribe",
                         [(input_dataset0, condition1, todescribe0, )])
def test_describe_array2(input_dataset, condition, todescribe):
    with pytest.raises(ValueError):
        ftddf.describe_array(input_dataset, todescribe=todescribe, condition=condition)

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

    a=ftddf.describe_array(testdata2, todescribe=['age'], condition=np.array(['f', 'f', 'm', 'm']))
    print(a)

    a=ftddf.describe_array(testdata2, todescribe=['age'])
    print(a)
