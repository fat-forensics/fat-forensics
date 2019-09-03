"""
Tets fatf.transparency.models.submodular_pick.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import abc

import pytest

import numpy as np

import fatf
import fatf.transparency.models.submodular_pick as ftms
import fatf.utils.data.datasets as fudd
from fatf.exceptions import IncompatibleExplainerError, IncorrectShapeError
from fatf.utils.testing.arrays import (BASE_NP_ARRAY, BASE_STRUCTURED_ARRAY,
                                       NOT_BASE_NP_ARRAY)

from sklearn.linear_model import LogisticRegression, Ridge


NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [0, 1, 0.07, 0.21]])

CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])

EXPLAINERS = [
    {'a': 0.1,
     'b': 0.1,
     'c': 0.1,
     'd': 0.1},
    {'a': 0.1,
     'b': 0.1},
    {'e': 0.1,
     'd': 0.1},
    {'a': 0.9,
     'b': 0.9,
     'c': 0.9,
     'd': 0.9}]


class DummyExplainer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dataset):
        self.dataset = dataset
    
    def explain_instance(self, instance):
        ind = np.where((self.dataset == instance).all(axis=1))
        i = ind[0][0]
        return self.explainers[i]

class DummyExplainer1(DummyExplainer):
    def __init__(self, dataset):
        self.dataset = dataset
        self.explainers = EXPLAINERS

class DummyExplainer2(DummyExplainer):
    def __init__(self, dataset):
        self.dataset = dataset
        self.explainers = EXPLAINERS[::-1]


def test_input_is_valid():
    """
    Tests :func:`fatf.transparency.models.submodular_pick._is_input_valid`.
    """
    class FakeExplainer():
        def __init__(self): pass
    
    class WorkingExplainer():
        def __init__(self): pass
        def explain_instance(self, x, labels=1): pass
    
    fake_explainer = FakeExplainer()
    working_explainer = WorkingExplainer()

    msg = ('The input dataset must be a 2-dimensional array.')
    with pytest.raises(IncorrectShapeError) as exin:
        ftms._input_is_valid(np.array([0,]), None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(ValueError) as exin:
        ftms._input_is_valid(NOT_BASE_NP_ARRAY, None, None, None)
    assert str(exin.value) == msg

    msg = ('sample_size must be an integer or None.')
    with pytest.raises(TypeError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, 'a', None)
    assert str(exin.value) == msg

    msg = ('sample_size must be a positive integer or None.')
    with pytest.raises(ValueError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, -1, None)
    assert str(exin.value) == msg

    msg = ('num_explanations must be an integer or None.')
    with pytest.raises(TypeError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, 1, 'a')
    assert str(exin.value) == msg

    msg = ('num_explanations must be a positive integer or None.')
    with pytest.raises(ValueError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, 1, -1)
    assert str(exin.value) == msg

    msg = ('explainer object must be method \'explain_instance\' which has '
           'exactly one required parameter. Other named parameters can be '
           'passed to the submodular pick method.')
    with pytest.raises(IncompatibleExplainerError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, fake_explainer, 1, 1)
    assert str(exin.value) == msg

    msg = ('sample_size must be larger or equal to num_explanations.')
    with pytest.raises(ValueError) as exin:
        ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, 1, 2)
    assert str(exin.value) == msg
    # All ok
    assert ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, 1, 1)
    assert ftms._input_is_valid(CATEGORICAL_STRUCT_ARRAY, working_explainer,
                                None, None)
    assert ftms._input_is_valid(NUMERICAL_NP_ARRAY, working_explainer, None, 1)


def test_submodular_pick():
    """
    Tests :func:`fatf.transparency.models.submodular_pick`.
    """
    fatf.setup_random_seed()

    explainer = DummyExplainer1(NUMERICAL_NP_ARRAY)
    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explainer, num_explanations=2)
    assert explanation_ind == [0, 2]
    assert explanations == [EXPLAINERS[0], EXPLAINERS[2]]

    explainer = DummyExplainer2(NUMERICAL_NP_ARRAY)
    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explainer, num_explanations=2)
    assert explanation_ind == [0, 1]
    assert explanations == [EXPLAINERS[3], EXPLAINERS[2]]

    explainer = DummyExplainer1(NUMERICAL_NP_ARRAY)
    msg = ('sample_size is larger than the number of sampels in the dataset. '
           'The whole dataset will be used.')
    with pytest.warns(UserWarning) as warning:
        explanations, explanation_ind = ftms.submodular_pick(
            NUMERICAL_NP_ARRAY, explainer, sample_size=100, num_explanations=1)
    assert len(warning) == 1
    assert str(warning[0].message)
    assert explanation_ind == [0]
    assert explanations == [EXPLAINERS[0]]

    explainer = DummyExplainer1(NUMERICAL_NP_ARRAY)
    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explainer, sample_size=1, num_explanations=1)
    assert explanation_ind == [2]
    assert explanations == [EXPLAINERS[2]]

    explainer = DummyExplainer1(NUMERICAL_NP_ARRAY)
    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explainer, sample_size=None, num_explanations=None)
    assert explanation_ind == [0, 2, 1, 3]
    assert explanations == [EXPLAINERS[0], EXPLAINERS[2], EXPLAINERS[1], 
                            EXPLAINERS[3]]

    explainer = DummyExplainer1(NUMERICAL_NP_ARRAY)
    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explainer, sample_size=2, num_explanations=None)
    assert explanation_ind == [3, 0]
    assert explanations == [EXPLAINERS[3], EXPLAINERS[0]]
