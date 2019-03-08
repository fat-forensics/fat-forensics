"""
The :mod:`fatf.core.datasets` contains the functions for loading in a few 
example datasets
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, List, Tuple, Union
import os
import csv

import numpy as np
import numpy.lib.recfunctions as rfn

import fatf.utils.validation as fuv


MODULE_PATH = os.path.dirname(__file__)

__all__ = ['load_iris', 'load_health_records', 'load_data']


def _validate_header(X: np.ndarray,
                     target: np.ndarray,
                     n_samples: int,
                     n_features: int,
                     target_names: np.ndarray) -> None:
    """
    Checks if reading in data is consistent.

    For details on valid header formatting see 
    :func:`fatf.utils.datasets.load_data`

    Parameters
    ----------
    X : numpy.ndarray
        Array read in from np.genfromtxt.
    target : numpy.ndarray
        Target variable specifying which class each sample in X belongs to.
    n_samples : int
        Number of samples expected in X.
    n_features : int
        Number of features expected in X.
    target_names : numpy.ndarray
        Class names corresponding to targets.
    
    Raises
    ------
    ValueError
        Number of samples in datset no consistent with header or number of
        features in dataset not consistent with header or number of classes
        not consistent with header.
    """
    if n_samples != X.shape[0]:
        raise ValueError('Number of samples in dataset not consistent with '
                         'header.')
    # Use len(X[0]) in case X is structured array.
    if n_features != len(X[0]):
        raise ValueError('Number of features in dataset not consistent with '
                         'header.')
    if (target_names.shape[0] and 
            target_names.shape[0] != np.unique(target).shape[0]):
        raise ValueError('Number of classes not consistent with header.')


def _get_header(filename: str) -> Tuple[int, int, np.ndarray]:
    """
    Reads first line of file and returns values about csv file.

    For details on valid header formatting see 
    :func:`fatf.utils.datasets.load_data`

    Parameters
    ----------
    filename : string
        Filename of the csv file.
    
    Raises
    ------
    ValueError
        Not enough arguments in header or first two arguments are not integers

    Returns
    -------
    n_samples : integer
        Number of sampels specified in header.
    n_features : integer
        Number of features specified in each sample.
    target_names : numpy.ndarray
        List of class_names specified in header.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
    if len(header) < 2:
        raise ValueError('Not enough arguments in header.')
    if not header[0].strip().isdigit():
        raise TypeError('{} is not a valid integer for number of samples '
                        'in the dataset'.format(header[0]))
    if not header[1].strip().isdigit():
        raise TypeError('{} is not a valid integer for number of samples '
                        'in the dataset'.format(header[1]))
    # TODO (Kacper): if header is not integer but isdigit = True
    n_samples = int(header[0])
    n_features = int(header[1])
    target_names = np.array(header[2:])
    return n_samples, n_features, target_names

def load_iris() -> Dict[str, np.ndarray]:
    """
    Loads in the IRIS dataset.

    .. [1] Fisher,R.A. "The use of multiple measurements in taxonomic problems" 
       Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions 
       to Mathematical Statistics" (John Wiley, NY, 1950). 

    Returns
    -------
    data: Dict[string, numpy.ndarray]
        See :func:`fatf.utils.datasets.load_data` for data format.
    """
    filename = os.path.join(MODULE_PATH, 'data', 'iris.csv')
    feature_names = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)']
    data = load_data(filename, dtype=np.float32, 
                     feature_names=feature_names)
    return data


def load_health_records() -> Dict[str, np.ndarray]:
    """
    Loads in data generated with Faker which is a structued np.ndarray 
    containing multiple dtypes.

    .. _Faker: https://github.com/joke2k/faker

    Returns
    -------
    data: Dict[string, numpy.ndarray]
        See :func:`fatf.utils.datasets.load_data` for data format.
    """
    filename = os.path.join(os.path.join(MODULE_PATH, 'data'), 
                            'health_records.csv')
    feature_names = ['name', 'email', 'age', 'weight', 'gender', 
                     'zipcode', 'diagnosis', 'dob']
    types = ['<U16', '<U25', '<i4', '<i4', '<U10', '<U6', 
             '<U6', '<U16']
    data = load_data(filename, dtype=list(zip(feature_names, types)),
                     feature_names=feature_names)
    return data


def load_data(filename: str, 
              dtype: Union[np.dtype,List[Tuple[str,str]], 
                           List[Tuple[str, np.dtype]]] = None,
              feature_names : List[str] = None
              ) -> Dict[str, np.ndarray]:
    """
    Loads dataset from file.

    File must have header in the first row in format
    [n_samples, n_features, class1_name, class2_name, ....]
    Then the rest of the csv will be treated as input data, with
    the last column of each row being a target variable. Types will
    be inferred if dtype=None and a structured array will be 
    returned unless all columns have the same inferred type.

    Parameters
    ----------
    filename : string
        Filename of csv to read in.
    dtype : Union[numpy.dtype,List[Tuple[string, string]], 
                  List[Tuple[string, numpy.dtype]]]
        dtypes to read filename as. Defaults to None where types
        will be inferred. In the case that the user wants to provide
        a list of dtypes to read in file as structured array with
        specific types.
    feature_names : List[string]
        List of names to be used as feature names. Defaults to None
        where feature names will just be 'feature0' ect..

    Raises
    ------
    ValueError
        Incorrect number of dtypes given.

    Returns
    -------
    data: Dict[string, numpy.ndarray]
        Dictionary with keys 'data', 'target', 'target_names',
        'feature_names' storing relevant information
    """
    n_samples, n_features, target_names = _get_header(filename)
    if isinstance(dtype, list):
        if len(dtype) == n_features + 1:
            if dtype[-1][0] != 'target':
                raise ValueError('If list of dtypes is given, the last one '
                                 'must be called `target`.')
        elif len(dtype) == n_features:
            dtype = dtype + [('target', '<i8')]
        else:
            raise ValueError('Incorrect number of dtypes given.')
    X = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                        dtype=dtype, encoding=None)
    if fuv.is_structured_array(X):
        # If np.genfromtxt infers structured array but user did 
        # not define dtype
        if dtype is None:
            target = X[X.dtype.names[-1]]
            X = rfn.drop_fields(X, X.dtype.names[-1])
        else:
            target = X['target']
            X = rfn.drop_fields(X, 'target')
    else:
        target = X[:, -1]
        X = np.delete(X, -1, 1)
    _validate_header(X, target, n_samples, n_features, target_names)
    if feature_names is None:
        feature_names = ['feature_{}'.format(i) for i in 
                                  range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise ValueError('Incorrect number of feature names given.')
        else:
            if fuv.is_structured_array(X):
                X.dtype.names = feature_names
    data = {'data': X,
            'target': target,
            'target_names': target_names,
            'feature_names': np.array(feature_names)}
    return data
