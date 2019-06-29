"""
The :mod:`fatf.utils.data.datasets` module holds examples of data sets.

The iris data set is returned as a classic numpy array, whereas the health
records data set is a structured numpy array.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import csv
import os

from typing import Dict, List, Tuple, Union

import numpy as np

import fatf.utils.tools as fut
import fatf.utils.array.validation as fuav

__all__ = ['load_data', 'load_health_records', 'load_iris']

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_14 = fut.at_least_verion([1, 14], _NUMPY_VERSION)

_DATA_PATH = os.path.join(os.path.dirname(__file__), 'datasets')


def _validate_data_header(X: np.ndarray, y: np.ndarray, n_samples: int,
                          n_features: int, y_names: np.ndarray) -> bool:
    """
    Checks if read-in data are consistent with their csv header.

    For details on valid header formatting see the
    :func:`fatf.utils.datasets.load_data` documentation.

    Parameters
    ----------
    X : numpy.ndarray
        Array read in from ``numpy.genfromtxt``.
    y : numpy.ndarray
        Target variable indicating which class each sample in ``X`` belongs to.
    n_samples : integer
        Number of samples expected in ``X`` and ``y``.
    n_features : integer
        Number of features expected in ``X``.
    y_names : numpy.ndarray
        Unique class names of the target variable ``y``.

    Raises
    ------
    ValueError
        The number of samples in ``X`` and ``y`` or the number of features in
        the dataset ``X`` is not consistent with the header. Also, raised when
        the number of unique classes in ``y`` is not consistent with the
        header.

    Returns
    -------
    is_consistent : boolean
        True if the header is consistent with the data, False otherwise.
    """
    # pylint: disable=invalid-name
    assert fuav.is_2d_array(X), 'X has to be a 2-dimensional array.'
    assert fuav.is_1d_array(y), 'y has to be a 1-dimensional array.'
    assert fuav.is_1d_array(y_names), 'y_names must be a 1-dimensional array.'

    is_consistent = False
    if X.shape[0] != n_samples:
        raise ValueError('The number of samples in the dataset is not '
                         'consistent with the header.')
    # Use len(X[0]) in case X is structured array.
    if len(X[0]) != n_features:
        raise ValueError('The number of features in the dataset is not '
                         'consistent with the header.')
    if y.shape[0] != n_samples:
        raise ValueError('The number of labels (target variables) is not '
                         'consistent with the header.')
    if y_names.shape[0]:
        if y_names.shape[0] != np.unique(y).shape[0]:
            raise ValueError('The number of classes is not consistent with '
                             'the header.')

    is_consistent = True
    return is_consistent


def _get_data_header(file_path: str) -> Tuple[int, int, np.ndarray]:
    """
    Reads the first line of a csv file and returns its header (the first row).

    For details on valid header formatting see the
    :func:`fatf.utils.datasets.load_data` documentation.

    Parameters
    ----------
    file_path : string
        Path to the csv data file.

    Raises
    ------
    ValueError
        The header is too short (expecting at least 2 elements) or the first
        two header elements are not integers.

    Returns
    -------
    n_samples : integer
        The expected number of samples in the dataset.
    n_features : integer
        The expected number of features in the dataset.
    target_names : numpy.ndarray
        A list of unique class (target) names expected in the dataset.
    """
    with open(file_path, 'r') as file_object:
        reader = csv.reader(file_object, delimiter=',')
        header = next(reader)

    if len(header) < 2:
        raise ValueError('The header is too short. Expecting at least 2 '
                         'entries, found {}.'.format(len(header)))

    h_zero = header[0].strip()
    h_one = header[1].strip()
    h_rest = [i.strip() for i in header[2:]]

    if not h_zero.isdecimal():
        raise TypeError('{} is not a valid integer. The number of samples in '
                        'the dataset has to be expressed as an '
                        'integer.'.format(h_zero))
    if not h_one.isdecimal():
        raise TypeError('{} is not a valid integer. The number of features in '
                        'the dataset has to be expressed as an '
                        'integer.'.format(h_one))

    n_samples = int(h_zero)
    n_features = int(h_one)
    target_names = np.array(h_rest)
    return n_samples, n_features, target_names


def load_data(file_path: str,
              dtype: Union[None, type, np.dtype, str, List[Tuple[str, str]],
                           List[Tuple[str, np.dtype]]] = None,
              feature_names: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Loads a dataset from a file.

    The dataset file must be formatted in the *comma separated value* (*csv*)
    standard with ``,`` used as the delimiter. The first row of the file must
    be a header formatted as follows:
    ``n_samples,n_features,class_name_1,class_name_2,...``, for example
    ``150,5,red,green,blue,black`` indicates that there are 150 data points,
    with 5 features and 4 possible classes: red, green, blue and black. The
    classes should be given in an order that matches the lexicographical
    ordering of the unique class values. For example, given that the class
    values in the data are: 3, 2, 4 and 1 the assignment would be: 1--red,
    2--green, 3--blue and 4--black. The rest of the csv file will be treated as
    a data array, with the last column being treated as the target (class)
    variable. The type of each column will be inferred if the ``dtype``
    parameter is set to ``None``, otherwise the array will be cased into the
    provided dtype. In case the columns in the data are of different types or
    the user-provided dtype defines the columns to be of multiple types a
    structured numpy array is used to represent the data.

    Parameters
    ----------
    file_path : string
        Path to the csv data file.
    dtype : Union[type, numpy.dtype, string, List[Tuple[string, string]], \
List[Tuple[string, type]], List[Tuple[string, numpy.dtype]]], \
optional (default=None)
        dtypes used to read the csv data. Defaults to None in which case the
        types will be inferred. The user can provide either a single type for
        the whole array (as a built-in Python type, numpy's dtype or a string
        representation of a numpy's dtype) or a list of tuples representing the
        name (string) and type (see above) of every column in the data array.
        In the latter case they user may choose to provide the list of types
        for the whole dataset, including the target column, or just the columns
        representing features.
    feature_names : List[string]
        List of strings representing the feature names. Defaults to None in
        which case features are given default names ('feature_0', etc.) or if a
        structured ``dtype`` parameter is provided the names given in the
        ``dtype`` parameter are used.

    Raises
    ------
    TypeError
        If provided, one of the feature names in the ``feature_names``
        parameter is not a string; the ``feature_names`` parameter is neither
        of the allowed types (None or a list); the first element of one of the
        ``dtype`` tuples is not a string or the ``dtype`` parameter is neither
        of the allowed types (None, a list of tuples, a built-in Python type,
        numpy's dtype or a string representation of a numpy's dtype).
    ValueError
        The number of feature names is inconsistent with the data header, the
        feature names are provided both in the ``feature_names`` and ``dtype``
        parameters, a tuple in the list of complex ``dtype``\\ s is
        malformatted, or the number of type definitions in the ``dtype``
        parameter is inconsistent with the number of features in the dataset.

    Returns
    -------
    data : Dict[string, numpy.ndarray]
        A dictionary representation of the dataset storing all the relevant
        information under the following keys: 'data', 'target', 'target_names',
        'feature_names'.
    """
    # pylint: disable=too-many-branches,too-many-statements
    if _NUMPY_1_14:  # pragma: no cover
        kwargs = dict(encoding=None)
    else:  # pragma: no cover
        kwargs = dict()
    n_samples, n_features, target_names = _get_data_header(file_path)

    if feature_names is None and not isinstance(dtype, list):
        feature_names = ['feature_{}'.format(i) for i in range(n_features)]
    elif isinstance(feature_names, list) and not isinstance(dtype, list):
        if len(feature_names) != n_features:
            raise ValueError('The number of feature names ({}) is '
                             'inconsistent with the number of features '
                             'encoded in the data header ({}).'.format(
                                 len(feature_names), n_features))
        for i, name in enumerate(feature_names):
            if not isinstance(name, str):
                raise TypeError('Element {} of the feature_names list is not '
                                'a string.'.format(i))
    elif feature_names is None and isinstance(dtype, list):
        pass
    elif isinstance(feature_names, list) and isinstance(dtype, list):
        raise ValueError('Feature names were provided both in feature_names '
                         'parameter and alongside types in dtype parameter. '
                         'Please choose one way to supply feature names.')
    else:
        raise TypeError('feature_names should either be None (to assign '
                        'default feature names or use the ones provided in '
                        'dtype) or a list of strings representing custom '
                        'feature names.')

    if dtype is None:
        target_dtype = None
    elif isinstance(dtype, (np.dtype, str, type)):
        dtype = np.dtype(dtype)
        target_dtype = dtype
    elif isinstance(dtype, list):
        feature_names = []
        typed_dtype = []
        for a_type in dtype:
            if not isinstance(a_type, tuple) or len(a_type) != 2:
                raise ValueError('Each dtype entry has to be a pair of two '
                                 'elements: column name (string) and column '
                                 'dtype (string, type or numpy.dtype). This '
                                 'entry does not comply: {}.'.format(a_type))
            if not isinstance(a_type[0], str):
                raise TypeError('Feature names included in the dtype '
                                'parameter should be strings. This feature '
                                'name does not comply: {}.'.format(a_type[0]))
            feature_names.append(a_type[0])
            feature_type = np.dtype(a_type[1])
            typed_dtype.append((a_type[0], feature_type))
        dtype = typed_dtype

        if len(dtype) == n_features + 1:
            target_dtype = dtype[-1][1]
            feature_names = feature_names[:-1]
            dtype = dtype[:-1]
        elif len(dtype) == n_features:
            target_dtype = None
        else:
            raise ValueError('The number of dtypes has to be either the same '
                             'as the number of features or the same as number '
                             'of features + 1 if the target dtype is '
                             'included. In the latter case, the name of the '
                             "last dtype must be 'target'.")
    else:
        raise TypeError('dtype should either be None (the types will be '
                        'inferred) or a list of tuples encoding each feature '
                        "name and type (in numpy's dtype notation).")

    data = np.genfromtxt(  # pylint: disable=unexpected-keyword-arg
        file_path,
        delimiter=',',
        skip_header=1,
        dtype=dtype,
        usecols=range(n_features),
        invalid_raise=True,
        **kwargs)
    if dtype is None and fuav.is_structured_array(data):
        data.dtype.names = feature_names

    target = np.genfromtxt(  # pylint: disable=unexpected-keyword-arg
        file_path,
        delimiter=',',
        skip_header=1,
        dtype=target_dtype,
        usecols=-1,
        invalid_raise=True,
        **kwargs)

    assert _validate_data_header(
        data, target, n_samples, n_features, target_names), \
        'The header must be valid for the read-in array.'

    data = {
        'data': data,
        'target': target,
        'target_names': target_names,
        'feature_names': np.array(feature_names)
    }
    return data


def load_iris() -> Dict[str, np.ndarray]:
    """
    Loads the IRIS dataset [FISHER1936]_.

    The dataset description can be found here_.

    .. [FISHER1936] Fisher,R.A. "The use of multiple measurements in taxonomic
       problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in
       "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).
    .. _here: https://archive.ics.uci.edu/ml/datasets/iris

    Returns
    -------
    data : Dict[string, numpy.ndarray]
        A dictionary with the dataset and its metadata. See
        :func:`fatf.utils.data.datasets.load_data` for the data format.
    """
    file_path = os.path.join(_DATA_PATH, 'iris.csv')
    data_dtype = np.float32
    feature_names = [
        'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)'
    ]
    data = load_data(file_path, dtype=data_dtype, feature_names=feature_names)
    return data


def load_health_records() -> Dict[str, np.ndarray]:
    """
    Loads in a fake health records dataset.

    The dataset contains a mixture of categorical and numerical columns
    generated with faker_. The data array is a structured numpy array with
    the following columns and types: ``'name'`` (string), ``'email'`` (string),
    ``'age'`` (integer), ``'weight'`` (integer), ``'gender'`` (string),
    ``'zipcode'`` (string), ``'diagnosis'`` (string) and ``'dob'`` (string) --
    date of birth. The target variable is binary (numerical) and encodes a
    medical treatment has succeeded: ``1`` is ``'success'`` and ``0`` is
    ``'fail'``.

    .. _faker: https://github.com/joke2k/faker

    Returns
    -------
    data : Dict[string, numpy.ndarray]
        A dictionary with the dataset and its metadata. See
        :func:`fatf.utils.data.datasets.load_data` for the data format.
    """
    file_path = os.path.join(_DATA_PATH, 'health_records.csv')
    feature_names = [
        'name', 'email', 'age', 'weight', 'gender', 'zipcode', 'diagnosis',
        'dob'
    ]
    feature_types = [
        '<U16', '<U25', '<i4', '<i4', '<U10', '<U6', '<U6', '<U16'
    ]
    dtype = list(zip(feature_names, feature_types))
    data = load_data(file_path, dtype=dtype)
    return data
