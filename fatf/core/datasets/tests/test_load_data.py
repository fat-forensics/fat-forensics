import os
import numpy as np
import pytest

from fatf.core.datasets.load_data import (load_health_records,
                                          load_iris,
                                          load_data)


def test_load_iris():
    # Check first, middle and last entry
    check_ind = np.array([0, 75, 149])
    true_data = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.6, 3. , 4.4, 1.4],
        [5.9, 3. , 5.1, 1.8]], dtype=np.float32)
    true_target = np.array([0, 1, 2], dtype=np.int32)
    target_names = np.array(['setosa', 'versicolor', 'virginica'])
    feature_names = np.array(['sepal length (cm)', 'sepal width (cm)',
                              'petal length (cm)', 'petal width (cm)'])

    iris_data = load_iris()
    assert iris_data['data'].shape == (150, 4)
    assert iris_data['target'].shape == (150, )
    assert np.array_equal(iris_data['data'][check_ind, :],true_data)
    assert np.array_equal(iris_data['target'][check_ind], true_target)
    assert np.array_equal(iris_data['target_names'], target_names)
    assert np.array_equal(iris_data['feature_names'], feature_names)


def test_load_health_records():
    # Check first, middle and last entry
    check_ind = np.array([0, 9, 20])
    dtypes = [('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), 
              ('gender', '<U10'), ('zipcode', '<U6'), ('diagnosis', '<U6'), 
              ('dob', '<U16')]
    true_data = np.array([
        ('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', 
         '03/06/2018'),
        ('Dean Campbell', 'michele18@hotmail.com', 62, 96, 'female', '2320', 'lung',
         '22/01/2009'),
        ('Susan Williams', 'smithjoshua@allen.com', 21, 42, 'male', '0203', 'lung',
         '15/11/2005')], dtype=dtypes)
    true_target = np.array([0, 0, 1])
    target_names = np.array(['success', 'fail'])
    feature_names = np.array([x for (x,y) in dtypes])
    
    cat_data = load_health_records()
    assert cat_data['data'].shape == (21, )
    assert cat_data['target'].shape == (21, )
    assert np.array_equal(cat_data['data'][check_ind], true_data)
    assert np.array_equal(cat_data['target'][check_ind], true_target)
    assert np.array_equal(cat_data['target_names'], target_names)
    assert np.array_equal(cat_data['feature_names'], feature_names)

def to_file(fname, string):
    with open(fname, 'w') as fp:
        fp.write(string)

def test_load_data(tmpdir):
    samples_err = 'Number of samples in dataset not consistent with header.'
    features_err = 'Number of features in dataset not consistent with header.'
    classes_err = 'Number of classes not consistent with header.'
    header_err = 'Not enough arguments in header.'
    number_err = 'Header of csv is in incorrect format'
    n_dtype_err = 'Incorrect number of dtypes given.'

    d = tmpdir.mkdir('temp')
    f = d.join('temp.csv')

    f.write('4,2,fail,pass\n1,2,0\n2,2,1', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath)
    assert str(exin.value) == samples_err

    f.write('2,2,fail,pass\n1,1\n2,0', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath)
    assert str(exin.value) == features_err

    f.write('2,1,fail\n1,0\n2,1', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath)
    assert str(exin.value) == classes_err

    f.write('2\n1,0\n,2,1', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath)
    assert str(exin.value) == header_err
    
    f.write('foo,bar\n1,0\n2,1', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath)
    assert str(exin.value) == number_err

    f.write('2,2,fail,pass\n1,2,0\n2,2,1', mode='w')
    with pytest.raises(ValueError) as exin:
        data = load_data(f.strpath, dtype=[('feat1', '<i4')])
    assert str(exin.value) == n_dtype_err

    f.write('2,2,fail,pass\n1,2,0\n2,2,1', mode='w')
    data = load_data(f.strpath, dtype=[('feat1', '<i4'), 
                                       ('feat2', '<i4'), 
                                       ('target', '<i4')])
    assert data['data'].shape == (2, )
    assert data['target'].shape == (2, )
    assert np.array_equal(data['data'][0], np.array((1, 2), 
                                           dtype=[('feat1', '<i4'),
                                                  ('feat2', '<i4')]))
    assert np.array_equal(data['feature_names'], 
                          np.array(['feature_0', 'feature_1']))
