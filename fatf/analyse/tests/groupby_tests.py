import numpy as np

import fatf.analyse.groupby as groupby


def get_dummy_data():
    ''' Make dummy dataset for testing '''
    data = np.array([[1, 2, 1],
                     [4, 5, 2],
                     [2, 2, 1],
                     [4, 4, 2]])
    col_names = ['f_1', 'f_2', 'class']
    return data, col_names


def test_splitter():
    ''' test splitter is splitting data '''
    data, col_names = get_dummy_data()
    split_col = groupby.splitter(data, col_names, 'f_1')
    correct_split = np.array([[1], [4], [2], [4]])
    assert (split_col == correct_split).all()


def test_mean_groupby():
    ''' test that function is being correctly applied to unique classes '''
    data, col_names = get_dummy_data()
    grouped_classes, grouped_means = groupby.apply_function(data, col_names, 'class', np.mean, 'f_1')
    correct_grouped_classes = np.array([1, 2])
    correct_grouped_means = np.array([1.5, 4])
    assert (grouped_classes == correct_grouped_classes).all()
    assert (grouped_means == correct_grouped_means).all()