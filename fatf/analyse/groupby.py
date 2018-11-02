'''
Implimentation similar to pandas groupby function, which applies a function to subgroups of a given column in the dataset

Author: Matt Clifford <mc15445@bristol.ac.uk>
License: new BSD
'''

import numpy as np


def splitter(data_frame, col_names, col_to_split):
    '''
    Splits off a column from the data frame.

    Args:
        data_frame (np.ndarray): 2-dimensional array of data values
        col_names (list): List containing strings of the column names for data_frame
        col_to_split (str): The column to split off the the data frame

    Returns:
        split_col (np.ndarray): Array containg column selected to be split from the data frame

    Raises:
        Exception: If col_to_split is not contained in col_names
        Exception: If col_names does not contain unique column names

    Examples:
        >>>data_frame = np.array([[1, 2, 1],
                                  [4, 5, 2],
                                  [2, 2, 1],
                                  [4, 4, 2]])
        >>>col_names = ['feature_1', 'feature_2', 'class']
        >>>groupby.splitter(data_frame, col_names, 'feature_1')
        array([[1],
               [4],
               [2],
               [4]])

    '''
    col_bool: bool = np.isin(col_names, col_to_split)     # boolean of the instances of col_to_split is
    num_of_cols: int = np.where(col_bool == True)[0].size
    if num_of_cols == 0:                            # if there are no columns with names col_to_split
        raise Exception('Column \'{0}\' is not in list of column names for the dataset'.format(col_to_split))
    elif num_of_cols > 1:                           # more than one instance of col_to_split
        raise Exception('Column names must be unique, there are {0} columns with the name \'{1}\''.format(num_of_cols, col_to_split))
    split_ind: int = np.where(col_bool == True)[0]       # get index of specified column
    split_col: float = data_frame[:, split_ind]            # split of desired column
    return split_col


def apply_function(data_frame, col_names, col_to_split, func_to_apply, col_to_apply_func):
    '''
    Applies function to the subgroups of a given column.

    Args:
        data_frame (np.ndarray): 2-dimensional array of data values
        col_names (list): List containing strings of the column names for data_frame
        col_to_split (str): The column to subgroup by
        func_to_apply (function): Function being applied to desired column
        col_to_apply_func (str): Name of column which func_to_apply is being applied to

    Returns:
        unique_classes (np.ndarray): Unique entries in col_to_split
        func_buckets (np.ndarray): Values returned from func_to_apply on col_to apply_func corresponding to the unique_classes found

    Examples:
        >>>data_frame = np.array([[1, 2, 1],
                                  [4, 5, 2],
                                  [2, 2, 1],
                                  [4, 4, 2]])
        >>>col_names = ['feature_1', 'feature_2', 'class']
        >>>groupby.apply_function(data_frame, col_names, 'class', np.mean, 'feature_1')
        (array([1, 2]), array([1.5, 4. ]))

    '''
    split_col: float = splitter(data_frame, col_names, col_to_split)
    unique_classes: float = np.unique(split_col)                       # get unique classes
    class_buckets: float = [0.0] * (len(unique_classes))
    func_buckets: float = [0.0] * (len(unique_classes))
    for single_class in unique_classes:
        # get indicies of where unique class are in main data_frame
        class_inds = np.where(split_col == single_class)[0]     # first array taken as split_col is nx1x1 dimensions
        current_bucket = np.argwhere(single_class == unique_classes)[0][0]
        class_buckets[current_bucket] = data_frame[class_inds]  # store all data from class into bucket
        func_col_only = splitter(data_frame[class_inds], col_names, col_to_apply_func)
        func_buckets[current_bucket] = func_to_apply(func_col_only)
    return unique_classes, np.array(func_buckets)
