import numpy as np

from fatf.exceptions import CustomValueError
from fatf.core.datasets.load_data import (load_generated_structured,
                                          load_iris)

def test_load_iris():
    # Check first, middle and last entry
    check_ind = np.array([0, 75, 149])
    true_data = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.6, 3. , 4.4, 1.4],
        [5.9, 3. , 5.1, 1.8]], dtype=np.float32)
    true_target = np.array([0, 1, 2], dtype=np.int32)
    target_names = ['setosa', 'versicolor', 'virginica']
    feature_names = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)']

    iris_data = load_iris()
    assert(iris_data['data'].shape == (150, 4))
    assert(iris_data['target'].shape == (150,))
    assert(np.array_equal(iris_data['data'][check_ind, :],true_data))
    assert(np.array_equal(iris_data['target'][check_ind], true_target))
    assert(iris_data['target_names'] == target_names)
    assert(iris_data['feature_names'] == feature_names)

def test_load_generated_structured():
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
    target_names = ['success', 'fail']
    feature_names = [x for (x,y) in dtypes]
    
    cat_data = load_generated_structured()
    assert(cat_data['data'].shape == (21,))
    assert(cat_data['target'].shape == (21,))
    assert(np.array_equal(cat_data['data'][check_ind], true_data))
    assert(np.array_equal(cat_data['target'][check_ind], true_target))
    assert(cat_data['target_names'] == target_names)
    assert(cat_data['feature_names'] == feature_names)

def test_load_data():
    #TODO: write tests for load_data function like testing when it should fail.
    return True
