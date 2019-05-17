"""
Tests functions responsible for loading data.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.array.validation as fuav
import fatf.utils.data.datasets as fudd
import fatf.utils.tools as fut

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_14 = fut.at_least_verion([1, 14], _NUMPY_VERSION)


def test_validate_data_header():
    """
    Tests :func:`fatf.utils.data.datasets._validate_data_header`.
    """
    assertion_2d = 'X has to be a 2-dimensional array.'
    assertion_1d_y = 'y has to be a 1-dimensional array.'
    assertion_1d_names = 'y_names must be a 1-dimensional array.'

    value_error_samples = ('The number of samples in the dataset is not '
                           'consistent with the header.')
    value_error_features = ('The number of features in the dataset is not '
                            'consistent with the header.')
    value_error_labels = ('The number of labels (target variables) is not '
                          'consistent with the header.')
    value_error_classes = ('The number of classes is not consistent with the '
                           'header.')

    one_d = np.array([1, 2, 0])
    two_d = np.array([[1, 2, 0], [2, 2, 1]])
    with pytest.raises(AssertionError) as exin:
        fudd._validate_data_header(one_d, two_d, 0, 0, two_d)
    assert str(exin.value).startswith(assertion_2d)
    with pytest.raises(AssertionError) as exin:
        fudd._validate_data_header(two_d, two_d, 0, 0, two_d)
    assert str(exin.value).startswith(assertion_1d_y)
    with pytest.raises(AssertionError) as exin:
        fudd._validate_data_header(two_d, one_d, 2, 3, two_d)
    assert str(exin.value).startswith(assertion_1d_names)

    X = np.array([[1, 2, 0], [2, 2, 1]])
    y = np.array([1, 0])
    y_too_long = np.array([1, 0, 1])
    n_samples = 2
    n_features = 3
    y_names = np.array(['fail', 'pass'])
    y_names_too_many = np.array(['fail', 'pass', 'ok'])
    with pytest.raises(ValueError) as exin:
        fudd._validate_data_header(X, y, n_samples - 1, n_features, y_names)
    assert str(exin.value) == value_error_samples
    with pytest.raises(ValueError) as exin:
        fudd._validate_data_header(X, y, n_samples, n_features - 1, y_names)
    assert str(exin.value) == value_error_features
    with pytest.raises(ValueError) as exin:
        fudd._validate_data_header(X, y_too_long, n_samples, n_features,
                                   y_names)
    assert str(exin.value) == value_error_labels
    with pytest.raises(ValueError) as exin:
        fudd._validate_data_header(X, y, n_samples, n_features,
                                   y_names_too_many)
    assert str(exin.value) == value_error_classes

    # All good
    assert fudd._validate_data_header(X, y, n_samples, n_features, y_names)

    # Empty names
    assert fudd._validate_data_header(X, y, n_samples, n_features,
                                      np.array([]))


def test_get_data_header(tmpdir):
    """
    Tests :func:`fatf.utils.data.datasets._get_data_header`.
    """
    value_error = ('The header is too short. Expecting at least 2 entries, '
                   'found 1.')
    type_error_samples = (' is not a valid integer. The number of samples '
                          'in the dataset has to be expressed as an integer.')
    type_error_features = (' is not a valid integer. The number of '
                           'features in the dataset has to be expressed as '
                           'an integer.')

    temp_dir = tmpdir.mkdir('fatf_temp')
    temp_file = temp_dir.join('dataset_temp.csv')
    temp_file_path = temp_file.strpath

    def write_to_temp(text):
        return temp_file.write(text, mode='w')

    def is_empty_array(array):
        return np.array_equal(np.array([]), array)

    write_to_temp('2\n1,0\n2,1')
    with pytest.raises(ValueError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value) == value_error

    write_to_temp('foo,2\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_samples)
    write_to_temp('1.0,2\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_samples)
    write_to_temp('1.5,2\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_samples)

    write_to_temp('2,bar\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_features)
    write_to_temp('2,2.0\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_features)
    write_to_temp('2,2.5\n1,0\n2,1')
    with pytest.raises(TypeError) as exin:
        fudd._get_data_header(temp_file_path)
    assert str(exin.value).endswith(type_error_features)

    write_to_temp('2,3\n1,0,1,0\n2,1,55,1')
    n_samples, n_features, targ_names = fudd._get_data_header(temp_file_path)
    assert n_samples == 2 and n_features == 3 and is_empty_array(targ_names)

    write_to_temp('2,3,test1\n1,0,1,0\n2,1,55,1')
    n_samples, n_features, targ_names = fudd._get_data_header(temp_file_path)
    assert (n_samples == 2 and n_features == 3
            and np.array_equal(np.array(['test1']), targ_names))

    write_to_temp('200,333,test1,test0\n1,0,1,0\n2,1,55,1')
    n_samples, n_features, targ_names = fudd._get_data_header(temp_file_path)
    assert (n_samples == 200 and n_features == 333
            and np.array_equal(np.array(['test1', 'test0']), targ_names))


def test_load_data(tmpdir):
    """
    Tests :func:`fatf.utils.data.datasets.load_data`.
    """
    value_error_feature_number = ('The number of feature names (3) is '
                                  'inconsistent with the number of features '
                                  'encoded in the data header (2).')
    type_error_feature_string = ('Element 1 of the feature_names list is not '
                                 'a string.')
    value_error_dual_names = ('Feature names were provided both in '
                              'feature_names parameter and alongside types in '
                              'dtype parameter. Please choose one way to '
                              'supply feature names.')
    type_error_feature_names = ('feature_names should either be None (to '
                                'assign default feature names or use the ones '
                                'provided in dtype) or a list of strings '
                                'representing custom feature names.')

    value_error_dtype_tuple = ('Each dtype entry has to be a pair of two '
                               'elements: column name (string) and column '
                               'dtype (string, type or numpy.dtype). This '
                               'entry does not comply: {}.')
    type_error_dtype_str = ('Feature names included in the dtype parameter '
                            'should be strings. This feature name does not '
                            'comply: {}.')
    value_error_dtype_number = ('The number of dtypes has to be either the '
                                'same as the number of features or the same '
                                'as number of features + 1 if the target '
                                'dtype is included. In the latter case, the '
                                "name of the last dtype must be 'target'.")
    type_error_dtype = ('dtype should either be None (the types will be '
                        'inferred) or a list of tuples encoding each feature '
                        "name and type (in numpy's dtype notation).")

    temp_dir = tmpdir.mkdir('fatf_temp')
    temp_file = temp_dir.join('dataset_temp.csv')
    temp_file_path = temp_file.strpath

    def write_to_temp(text):
        return temp_file.write(text, mode='w')

    simple_data = np.array([[1, 2], [2, 2]])
    simple_target = np.array([0, 1])
    simple_target_names = np.array(['fail', 'pass'])
    simple_feature_names_a = np.array(['feature_0', 'feature_1'])
    write_to_temp('2,2,fail,pass\n1,2,0\n2,2,1')

    # No feature names and incorrect type dtype
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, dtype=7)
    assert str(exin.value) == type_error_dtype

    # No feature names and no dtype
    loaded_data = fudd.load_data(temp_file_path)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.int
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.int
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], simple_feature_names_a)

    # No feature names and a simple dtype
    loaded_data = fudd.load_data(temp_file_path, dtype=np.float32)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float32
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float32
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], simple_feature_names_a)
    #
    loaded_data = fudd.load_data(temp_file_path, dtype='f')
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float32
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float32
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], simple_feature_names_a)
    #
    loaded_data = fudd.load_data(temp_file_path, dtype=float)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float64
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float64
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], simple_feature_names_a)

    # No feature names and a complex dtype -- excluding target
    my_dtype = [('a', np.float64), ('b', 'f')]
    my_names = np.array(['a', 'b'])
    my_data = np.array([(1., 2.), (2., 2.)], dtype=my_dtype)
    loaded_data = fudd.load_data(temp_file_path, dtype=my_dtype)
    assert fuav.is_structured_array(loaded_data['data'])
    assert np.array_equal(loaded_data['data'], my_data)
    for i in range(len(my_dtype)):
        assert loaded_data['data'].dtype.names[i] == my_dtype[i][0]
        assert loaded_data['data'].dtype[i] == my_dtype[i][1]
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.int
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names)
    # ...incorrect complex dtype
    my_dtype = [('a', np.float64), 'b']
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, dtype=my_dtype)
    assert str(exin.value) == value_error_dtype_tuple.format('b')
    my_dtype = [('a', np.float64), ('b', )]
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, dtype=my_dtype)
    assert str(exin.value) == value_error_dtype_tuple.format("('b',)")
    my_dtype = [('a', np.float64), (np.float64, 'b')]
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, dtype=my_dtype)
    e_format = "<class 'numpy.float64'>"
    assert str(exin.value) == type_error_dtype_str.format(e_format)
    my_dtype = [('a', np.float64)]
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, dtype=my_dtype)
    assert str(exin.value) == value_error_dtype_number

    # No feature names and a complex dtype -- including target
    my_dtype = [('a', np.float64), ('b', 'f'), ('targ', 'U1')]
    my_names = np.array(['a', 'b'])
    my_target = np.array(['0', '1'])
    my_data = np.array([(1., 2.), (2., 2.)], dtype=my_dtype[:-1])
    loaded_data = fudd.load_data(temp_file_path, dtype=my_dtype)
    assert fuav.is_structured_array(loaded_data['data'])
    assert np.array_equal(loaded_data['data'], my_data)
    for i in range(len(my_dtype[:-1])):
        assert loaded_data['data'].dtype.names[i] == my_dtype[i][0]
        assert loaded_data['data'].dtype[i] == my_dtype[i][1]
    assert np.array_equal(loaded_data['target'], my_target)
    assert loaded_data['target'].dtype == np.dtype('U1')
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names)

    ##########

    # feature_names supplied and incorrect dtype
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=['a', 'b'], dtype=7)
    assert str(exin.value) == type_error_dtype

    # feature_names supplied and no dtype
    my_names = ['a', 'b']
    my_names_np = np.array(my_names)
    loaded_data = fudd.load_data(temp_file_path, feature_names=my_names)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.int
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.int
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names_np)
    # ...wrong type
    wrong_type = ['a', 3]
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=wrong_type)
    assert str(exin.value) == type_error_feature_string
    # ...wrong quantity
    wrong_type = ['a', 'c', 'b']
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, feature_names=wrong_type)
    assert str(exin.value) == value_error_feature_number

    # feature_names supplied and a simple dtype
    my_names = ['a', 'b']
    my_names_np = np.array(my_names)
    loaded_data = fudd.load_data(
        temp_file_path, feature_names=my_names, dtype=np.float32)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float32
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float32
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names_np)
    #
    loaded_data = fudd.load_data(
        temp_file_path, feature_names=my_names, dtype='f')
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float32
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float32
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names_np)
    #
    loaded_data = fudd.load_data(
        temp_file_path, feature_names=my_names, dtype=float)
    assert np.array_equal(loaded_data['data'], simple_data)
    assert loaded_data['data'].dtype == np.float64
    assert np.array_equal(loaded_data['target'], simple_target)
    assert loaded_data['target'].dtype == np.float64
    assert np.array_equal(loaded_data['target_names'], simple_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_names_np)

    # feature_names supplied and a complex dtype -- excluding target
    my_names = ['a', 'b']
    my_types = [('a', int), ('b', int)]
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, feature_names=my_names, dtype=my_types)
    assert str(exin.value) == value_error_dual_names

    # feature_names supplied and a complex dtype -- including target
    my_names = ['a', 'b']
    my_types = [('a', int), ('b', float), ('targ', int)]
    with pytest.raises(ValueError) as exin:
        fudd.load_data(temp_file_path, feature_names=my_names, dtype=my_types)
    assert str(exin.value) == value_error_dual_names

    ##########

    # Incorrect feature_names supplied and incorrect dtype
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=4, dtype=2)
    assert str(exin.value) == type_error_feature_names

    # Incorrect feature_names supplied and no dtype
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=4)
    assert str(exin.value) == type_error_feature_names

    # Incorrect feature_names supplied and a simple dtype
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=4, dtype=np.int)
    assert str(exin.value) == type_error_feature_names

    # Incorrect feature_names supplied and a complex dtype
    my_names = ['a', 'b']
    my_types = [('a', int), ('b', float), ('targ', int)]
    with pytest.raises(TypeError) as exin:
        fudd.load_data(temp_file_path, feature_names=4, dtype=my_types)
    assert str(exin.value) == type_error_feature_names

    # A mixture of categorical and numerical feature and a categorical target
    string_type = 'U{}' if _NUMPY_1_14 else 'S{}'
    write_to_temp('2,2,fail,pass\n1,foo,pass\n2,bar,fail')
    my_feature_names = ['feature_0', 'feature_1']
    my_feature_names_np = np.array(my_feature_names)
    my_feature_dtype = [('feature_0', np.int),
                        ('feature_1', np.dtype(string_type.format('3')))]
    my_data = np.array([(1, 'foo'), (2, 'bar')], dtype=my_feature_dtype)
    my_target = np.array(['pass', 'fail'], dtype=string_type.format('4'))
    my_target_names = np.array(['fail', 'pass'])
    #
    loaded_data = fudd.load_data(temp_file_path)
    assert np.array_equal(loaded_data['data'], my_data)
    for i in range(len(my_feature_names)):
        assert loaded_data['data'].dtype.names[i] == my_feature_names[i]
        assert loaded_data['data'].dtype[i] == my_feature_dtype[i][1]
    assert np.array_equal(loaded_data['target'], my_target)
    assert loaded_data['target'].dtype == np.dtype(string_type.format('4'))
    assert np.array_equal(loaded_data['target_names'], my_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_feature_names_np)
    #
    my_feature_names = ['f0', 'f1']
    my_feature_names_np = np.array(my_feature_names)
    my_feature_dtype = [('f0', np.int),
                        ('f1', np.dtype(string_type.format('3')))]
    my_data = np.array([(1, 'foo'), (2, 'bar')], dtype=my_feature_dtype)
    loaded_data = fudd.load_data(
        temp_file_path, feature_names=my_feature_names)
    assert np.array_equal(loaded_data['data'], my_data)
    for i in range(len(my_feature_names)):
        assert loaded_data['data'].dtype.names[i] == my_feature_names[i]
        assert loaded_data['data'].dtype[i] == my_feature_dtype[i][1]
    assert np.array_equal(loaded_data['target'], my_target)
    assert loaded_data['target'].dtype == np.dtype(string_type.format('4'))
    assert np.array_equal(loaded_data['target_names'], my_target_names)
    assert np.array_equal(loaded_data['feature_names'], my_feature_names_np)


def test_load_iris():
    """
    Tests :func:`fatf.utils.data.datasets.load_iris`.
    """
    # Check the first, middle and last entry in the dataset.
    n_samples = 150
    n_features = 4
    check_ind = np.array([0, 75, 149])
    true_data = np.array([[5.1, 3.5, 1.4, 0.2],
                          [6.6, 3.0, 4.4, 1.4],
                          [5.9, 3.0, 5.1, 1.8]])  # yapf: disable
    true_target = np.array([0, 1, 2])
    target_names = np.array(['setosa', 'versicolor', 'virginica'])
    feature_names = np.array([
        'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)'
    ])

    iris_data = fudd.load_iris()
    assert not fuav.is_structured_array(iris_data['data'])
    assert iris_data['data'].shape == (n_samples, n_features)
    assert iris_data['target'].shape == (n_samples, )
    assert np.array_equal(iris_data['target_names'], target_names)
    assert np.array_equal(iris_data['feature_names'], feature_names)

    assert np.isclose(iris_data['data'][check_ind, :], true_data).all()
    assert np.isclose(iris_data['target'][check_ind], true_target).all()


def test_load_health_records():
    """
    Tests :func:`fatf.utils.data.datasets.load_health_records`.
    """
    # Check the first, middle and last entry in the dataset.
    n_samples = 21
    n_features = 8
    check_ind = np.array([0, 9, 20])
    dtypes = [('name', '<U16'), ('email', '<U25'), ('age', '<i4'),
              ('weight', '<i4'), ('gender', '<U10'), ('zipcode', '<U6'),
              ('diagnosis', '<U6'), ('dob', '<U16')]
    true_data = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52,
                           'female', '1121', 'cancer', '03/06/2018'),
                          ('Dean Campbell', 'michele18@hotmail.com', 62, 96,
                           'female', '2320', 'lung', '22/01/2009'),
                          ('Susan Williams', 'smithjoshua@allen.com', 21, 42,
                           'male', '0203', 'lung', '15/11/2005')],
                         dtype=dtypes)
    true_target = np.array([0, 0, 1])
    target_names = np.array(['fail', 'success'])
    feature_names = np.array([x for (x, y) in dtypes])

    cat_data = fudd.load_health_records()
    assert fuav.is_structured_array(cat_data['data'])
    assert cat_data['data'].shape == (n_samples, )
    assert len(cat_data['data'].dtype.names) == n_features
    assert cat_data['target'].shape == (n_samples, )
    assert np.array_equal(cat_data['target_names'], target_names)
    assert np.array_equal(cat_data['feature_names'], feature_names)

    assert np.array_equal(cat_data['data'][check_ind], true_data)
    assert np.array_equal(cat_data['target'][check_ind], true_target)
