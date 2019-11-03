"""
.. versionadded:: 0.0.2

The :mod:`fatf.utils.testing.transparency` module holds transparency testing
assets.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

__all__ = ['LABELS',
           'NUMERICAL_NP_ARRAY',
           'NUMERICAL_STRUCT_ARRAY',
           'CATEGORICAL_NP_ARRAY',
           'CATEGORICAL_STRUCT_ARRAY',
           'MIXED_ARRAY',
           'InvalidModel',
           'NonProbabilisticModel',
           'is_explanation_equal_dict',
           'is_explanation_equal_list']  # yapf: disable

ExplanationList = Dict[str, List[Tuple[str, Union[Iterable[float], float]]]]
ExplanationDict = Dict[str,
                       Dict[Union[str, int], Union[Iterable[float], float]]]

# yapf: disable
LABELS = np.array([2, 0, 1, 1, 0, 2])

NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', int), ('b', int), ('c', float), ('d', float)])

CATEGORICAL_NP_ARRAY = np.array(
    [['a', 'b', 'c'], ['a', 'f', 'g'], ['b', 'c', 'c'],
     ['b', 'f', 'c'], ['a', 'f', 'c'], ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'), ('a', 'f', 'g'), ('b', 'c', 'c'),
     ('b', 'f', 'c'), ('a', 'f', 'c'), ('a', 'b', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])

MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'), (0, 'f', 0.03, 'bb'), (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'), (0, 'c', 0.36, 'b'), (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
# yapf: enable


class InvalidModel(object):
    """
    An invalid model class -- it does not implement a ``predict_proba`` method.

    .. versionadded:: 0.0.2
    """

    # pylint: disable=useless-object-inheritance,too-few-public-methods

    def __init__(self):
        """
        Invalid initialisation.
        """

    def fit(self, data, targets):
        """
        Invalid fit.
        """


class NonProbabilisticModel(InvalidModel):
    """
    A model that is not probabilistic -- no ``predict_proba`` function.

    .. versionadded:: 0.0.2
    """

    def __init__(self, prediction_function):
        """
        Non-probabilistic initialisation.
        """
        super().__init__()
        self.prediction_function = prediction_function

    def predict(self, data):
        """
        Non-probabilistic predict.
        """
        return self.prediction_function(data)


def is_explanation_equal_list(dict1: ExplanationList,
                              dict2: ExplanationList,
                              atol: float = 1e-1) -> bool:
    """
    Tests if the two dictionaries of a given structure are equal.

    .. versionadded:: 0.0.2

    The both of the input parameters must be a dictionary with string keys and
    list values. The latter is composed of 2-tuples of strings and floats or
    float iterables.

    The keys in the dictionary and the tuples must match exactly, while the
    floats only need to be approximately equal. The ordering of the tuples in
    the list does not need to be the same.

    Parameters
    ----------
    dict1 : Dictionary[string,\
List[Tuple[string, Union[Iterable[float], float]]]
        The first dictionary to be compared.
    dict2 : Dictionary[string,\
List[Tuple[string, Union[Iterable[float], float]]]
        The second dictionary to be compared.
    atol : float, optional (default=0.1)
        The absolute tolerance between each two matching numbers in the inner
        dictionaries.

    Returns
    -------
    equal : boolean
        ``True`` if the dictionaries are the same, ``False`` otherwise.
    """
    if set(dict1.keys()) == set(dict2.keys()):
        equal = True
        for key in dict1:
            val1 = sorted(dict1[key])
            val2 = sorted(dict2[key])

            if len(val1) != len(val2):
                equal = False
                break

            for i, val1_i in enumerate(val1):
                if val1_i[0] != val2[i][0]:
                    equal = False
                    break
                is_close = np.allclose(
                    val1_i[1], val2[i][1], atol=atol, equal_nan=True)
                if not is_close:
                    equal = False
                    break

            if not equal:
                break
    else:
        equal = False
    return equal


def is_explanation_equal_dict(dict1: ExplanationDict,
                              dict2: ExplanationDict,
                              atol: float = 1e-3) -> bool:
    """
    Tests if the two dictionaries of a given structure are equal.

    .. versionadded:: 0.0.2

    The both of the input parameters must be a dictionary with string keys and
    dictionary values. The latter one has strings or integers as its keys and
    numbers or number iterables (to be compared with given absolute tolerance)
    as its values.

    The keys in the outer and inner dictionaries must match exactly, while the
    numbers only need to be approximately (``atol``) equal.

    Parameters
    ----------
    dict1 : Dictionary[string,\
Dictionary[Union[string, integer], Union[Iterable[float], float]]]
        The first dictionary to be compared.
    dict2 : Dictionary[string,\
Dictionary[Union[string, integer], Union[Iterable[float], float]]]
        The second dictionary to be compared.
    atol : float, optional (default=0.001)
        The absolute tolerance between each two matching numbers in the inner
        dictionaries.

    Returns
    -------
    equal : boolean
        ``True`` if the dictionaries are the same, ``False`` otherwise.
    """
    if set(dict1.keys()) == set(dict2.keys()):
        equal = True
        for outer_key in dict1:
            inner_dict1 = dict1[outer_key]
            inner_dict2 = dict2[outer_key]

            if set(inner_dict1.keys()) == set(inner_dict2.keys()):
                for inner_key in inner_dict1.keys():
                    val1 = inner_dict1[inner_key]
                    val2 = inner_dict2[inner_key]

                    if not np.allclose(val1, val2, atol=atol, equal_nan=True):
                        equal = False
                        break
            else:
                equal = False
                break

            if not equal:
                break
    else:
        equal = False
    return equal
