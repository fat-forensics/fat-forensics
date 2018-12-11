"""
The :mod:`fatf.core.datasets` contains the functions for 
loading in a few example datasets
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: BSD Clause 3

from typing import Dict, List, Tuple, Union
import os
import csv

import numpy as np
from numpy.lib.recfunctions import drop_fields

from fatf.exceptions import CustomValueError


MODULE_PATH = os.path.dirname(__file__)

def _check_input(X: np.ndarray,
                 target: np.ndarray,
                 n_samples: int,
                 n_features: int,
                 target_names: List[str]) -> None:
    """Checks if reading in data is consistent.

    Args
    ----
    X : np.ndarray
        Array read in from np.genfromtxt.
    target : np.ndarray
        Target variabel specifying which class each sample in X
        belongs to.
    n_samples : int
        Number of samples expected in X.
    n_features : int
        Number of features expected in X.
    target_names : List[str]
        Class names corresponding to targets.
    """
    if n_samples != X.shape[0]:
        raise CustomValueError('Number of samples in dataset not '
                               'consistent with header.')
    if n_features != len(X[0]):
        raise CustomValueError('Number of features in dataset not '
                               'consistent with header.')
    if len(target_names) != np.unique(target).shape[0]:
        raise CustomValueError('Number of classes not consistent '
                               'with header.')
    
def _get_header(filename: str) -> Tuple[int, int, List[str]]:
    """Reads first line of file and returns values about csv file.

    Args
    ----
    filename : str
        Filename of the csv file.
    
    Returns
    ----
    n_samples : int
        Number of sampels specified in header.
    n_features : int
        Number of features specified in each sample.
    target_names : List[str]
        List of class_names specified in header.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
    if len(header) < 4:
        raise CustomValueError('Not enough arguments in header.')
    try:
        n_samples = int(header[0])
        n_features = int(header[1])
    except ValueError as e:
        raise CustomValueError('Header of csv in incorrect '
                               'format.')
    target_names = header[2:]
    return (n_samples, n_features, target_names)

def load_iris() -> Dict[str, Union[np.ndarray, List[str]]]:
    """Function for loading in the IRIS dataset.

    Returns
    ----
    data: Dict[str, np.ndarray]
        Dictionary with keys 'data', 'target', 'target_names',
        'feature_names' storing relevant information
    """
    filename = os.path.join(os.path.join(MODULE_PATH, 'data'), 
                            'iris.csv')
    feature_names = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)']
    data = load_data(filename, dtype=np.float32, 
                     feature_names=feature_names)
    return data

    
def load_generated_structured() -> Dict[str, Union[np.ndarray, 
                                                  List[str]]]:
    """Function for loading in data generated with Faker which is
    a structued np.ndarray containing multiple dtypes.

    Returns
    ----
    data: Dict[str, np.ndarray]
        Dictionary with keys 'data', 'target', 'target_names',
        'feature_names' storing relevant information
    """
    filename = os.path.join(os.path.join(MODULE_PATH, 'data'), 
                            'generated_structured.csv')
    feature_names = ['name', 'email', 'age', 'weight', 'gender', 
                     'zipcode', 'diagnosis', 'dob']
    types = ['<U16', '<U25', '<i4', '<i4', '<U10', '<U6', 
             '<U6', '<U16']
    data = load_data(filename, dtype=list(zip(feature_names, types)),
                     feature_names=feature_names)
    return data

def load_data(filename: str, 
              dtype: Union[np.dtype,List[Tuple[str,str]]] = None,
              feature_names : List[str] = None
              ) -> Dict[str, Union[np.ndarray, List[str]]]:
    """Functions for loading in dataset.

    File must have header in the first row in format
    [n_samples, n_features, class1_name, class2_name, ....]
    Then the rest of the csv will be treated as input data, with
    the last column of each row being a target variable. Types will
    be inferred if dtype=None and a structured array will be 
    returned unless all columns have the same inferred type.

    Args
    ----
    filename : str
        Filename of csv to read in.
    dtype : Union[np.dtype,List[Tuple[str, str]]]
        Dtypes to read filename as. Defaults to None where types
        will be inferred. In the case that the user wants to provide
        a list of dtypes to read in file as structured array with
        specific types.
    feature_names : List[str]
        List of names to be used as feature names. Defaults to None
        where feature names will just be 'feature0' ect..
    Returns
    ----
    data: Dict[str, np.ndarray]
        Dictionary with keys 'data', 'target', 'target_names',
        'feature_names' storing relevant information
    """
    n_samples, n_features, target_names = _get_header(filename)
    if isinstance(dtype, list):
        print(dtype)
        if len(dtype) != 0:
            dtype = dtype + [('target', '<i4')]
    X = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                        dtype=dtype)
    if len(X.dtype) == 0:
        target = X[:, -1].astype(np.int32)
        X = X[:, 0:-1]
    else:
        target = X['target']
        X = drop_fields(X, 'target')
    if feature_names is None:
        feature_names = ['feature{}'.format(i) for i in 
                         range(0, len(n_features))]
    data = {'data': X,
            'target': target,
            'target_names': target_names,
            'feature_names': feature_names}
    return data
