"""
Tests Blimey class.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Tuple, Union, Optional, List

import pytest

import numpy as np

import fatf

import fatf.transparency.models.blimey as ftmb
import fatf.utils.models as fum
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretization as fudd
import fatf.transparency.models.submodular_pick as ftmsp

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

from sklearn.linear_model import Ridge

Index = Union[int, str]

ONE_D_ARRAY = np.array([0, 4, 3, 0])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])

NUMERICAL_NP_BLIMEY = {
    'Class A': {'A <= 0.00': -0.3665,
                'B <= 0.00': 0.1791,
                '0.07 < C <= 0.22': -0.0152,
                '0.58 < D <= 0.79': 0.0821},
    'Class B': {'A <= 0.00': 0.1489,
                'B <= 0.00': -0.0791,
                '0.07 < C <= 0.22': 0.0546,
                '0.58 < D <= 0.79': 0.0073},
    'Class C': {'A <= 0.00': 0.2177,
                'B <= 0.00': -0.1000,
                '0.07 < C <= 0.22': -0.0393,
                '0.58 < D <= 0.79': -0.0894}}

NUMERICAL_NP_BLIMEY_CAT= {
    'Class A': {'A': -0.1824,
                'B <= 0.00': 0.1318,
                '0.07 < C <= 0.22': 0.0526,
                '0.58 < D <= 0.79': 0.0276},
    'Class B': {'A': -0.0425,
                'B <= 0.00': -0.0670,
                '0.07 < C <= 0.22': -0.0730,
                '0.58 < D <= 0.79': -0.0237},
    'Class C': {'A': 0.2249,
                'B <= 0.00': -0.0648,
                '0.07 < C <= 0.22': 0.0204,
                '0.58 < D <= 0.79': -0.0039}}

class InvalidModel(object):
    """
    Tests for exceptions when a model lacks the ``predict_proba`` method.
    """

    def __init__(self):
        """
        Initialises not-a-model.
        """
        pass

    def fit(self, X, y):
        """
        Fits not-a-model.
        """
        return X, y  # pragma: nocover

def _is_explanation_equal(dict1: Dict[str, Dict[Index, np.float64]],
                          dict2: Dict[str, Dict[Index, np.float64]]) -> bool:
    """
    """
    equal = True

    if set(dict1.keys()) == set(dict2.keys()):
        for key in dict1.keys():
            class_exp1 = dict1[key]
            class_exp2 = dict2[key]
            if set(class_exp1.keys()) == set(class_exp2.keys()):
                for feature in class_exp1.keys():
                    feat_vals1 = class_exp1[feature]
                    feat_vals2 = class_exp2[feature]
                    if not np.isclose(feat_vals1, feat_vals2, atol=1e-2):
                        equal = False
            else:
                equal = False
    else:
        equal = False
    return equal


def test_input_is_valid():
    """
    Tests :func:`fatf.transparency.models.blimey._is_input_valid`.
    """
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftmb._input_is_valid(ONE_D_ARRAY, None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(np.array([[0, None], [0, 8]]), None, None, None,
                             None, None)
    assert str(exin.value) == msg

    msg = ('This functionality requires the global model to be capable of '
           'outputting probabilities via predict_proba method.')
    model = InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, None)
    assert str(exin.value) == msg

    msg = ('This functionality requires the local model to be capable of '
           'outputting probabilities via predict_proba method.')
    model = fum.KNN()
    with pytest.raises(IncompatibleModelError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, InvalidModel)
    assert str(exin.value) == msg

    msg = ('The following indices are invalid for the input dataset: '
           '{}.'.format(np.array(['a'])))
    with pytest.raises(IndexError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, ['a'])
    assert str(exin.value) == msg

    msg = ('The categorical_indices parameter must be a Python list or None.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, 'a')
    assert str(exin.value) == msg

    msg = ('The augmentor object must inherit from abstract class '
           'fatf.utils.augmentation.Augmentation.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
                NUMERICAL_NP_ARRAY, InvalidModel, fudd.QuartileDiscretizer,
                ftmsp.SKLearnExplainer, model, fum.KNN)
    assert str(exin.value) == msg

    msg = ('The discretizer object must inherit from abstract class '
           'fatf.utils.discretization.Discretization.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, InvalidModel,
            ftmsp.SKLearnExplainer, model, fum.KNN)
    assert str(exin.value) == msg

    msg = 'The class_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, class_names='a')
    assert str(exin.value) == msg

    msg = ('The class_name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, class_names=[0])
    assert str(exin.value) == msg

    msg = 'The feature_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, feature_names='a')
    assert str(exin.value) == msg
    
    msg = ('The length of feature_names must be equal to the number of '
           'features in the dataset.')
    with pytest.raises(ValueError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, feature_names=['a'])
    assert str(exin.value) == msg

    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling, fudd.QuartileDiscretizer,
            ftmsp.SKLearnExplainer, model, fum.KNN, feature_names=[0, 1, 2, 3])
    assert str(exin.value) == msg


class TestBlimey():
    """
    Tests :class:`fatf.transparency.models.blimey.Blimey`.
    """
    knn_numerical = fum.KNN()
    knn_numerical.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    numerical_blimey = ftmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        discretizer=fudd.QuartileDiscretizer,
        explainer=ftmsp.SKLearnExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=['Class A', 'Class B', 'Class C'],
        feature_names=['A', 'B', 'C', 'D'])

    numerical_blimey_cat = ftmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        discretizer=fudd.QuartileDiscretizer,
        explainer=ftmsp.SKLearnExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=['Class A', 'Class B', 'Class C'],
        feature_names=['A', 'B', 'C', 'D'],
        categorical_indices=[0])

    def test_blimey_class_init(self):
        """
        Tests :class:`fatf.transparency.models.blimey.Blimey` class init.
        """
        return True

    def test_explain_instance(self):
        """
        Tests :func:`fatf.transparency.models.blimey.Blimey.explain_instance`.
        """
        fatf.setup_random_seed()
        exp = self.numerical_blimey.explain_instance(NUMERICAL_NP_ARRAY[0])
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY)

        exp = self.numerical_blimey_cat.explain_instance(NUMERICAL_NP_ARRAY[0])
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_CAT)
