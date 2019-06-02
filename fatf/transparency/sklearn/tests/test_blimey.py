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
import fatf.utils.explainers as fue
import fatf.transparency.sklearn.linear_model as ftslm

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
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c'],
    ['b', 'f', 'c'],
    ['a', 'f', 'c'],
    ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c'),
     ('b', 'f', 'c'),
     ('a', 'f', 'c'),
     ('a', 'b', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])

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

NUMERICAL_NP_BLIMEY_CAT = {
    'Class A': {'A = 0': -0.1824,
                'B <= 0.00': 0.1318,
                '0.07 < C <= 0.22': 0.0526,
                '0.58 < D <= 0.79': 0.0276},
    'Class B': {'A = 0': -0.0425,
                'B <= 0.00': -0.0670,
                '0.07 < C <= 0.22': -0.0730,
                '0.58 < D <= 0.79': -0.0237},
    'Class C': {'A = 0': 0.2249,
                'B <= 0.00': -0.0648,
                '0.07 < C <= 0.22': 0.0204,
                '0.58 < D <= 0.79': -0.0039}}

NUMERICAL_NP_BLIMEY_NO_DISC = {
    'Class A': {'A': 0.246,
                'B': -0.228,
                'C': -0.038,
                'D': 0.050},
    'Class B': {'A': -0.096,
                'B': 0.141,
                'C': 0.071,
                'D': -0.050},
    'Class C': {'A': -0.149,
                'B': 0.087,
                'C': -0.032,
                'D': 0.000}}

NUMERICAL_STRUCT_BLIMEY = {
    'Class A': {'A = 0': -0.308,
                'B = 0': 0.168,
                '0.07 < C <= 0.22': 0.0390,
                '0.58 < D <= 0.79': -0.006},
    'Class B': {'A = 0': -0.030,
                'B = 0': -0.087,
                '0.07 < C <= 0.22': 0.055,
                '0.58 < D <= 0.79': 0.018},
    'Class C': {'A = 0': 0.338,
                'B = 0': -0.080,
                '0.07 < C <= 0.22': -0.094,
                '0.58 < D <= 0.79': -0.012}}

CATEGORICAL_NP_BLIMEY = {}
CATEGORICAL_STRUCT_BLIMEY = {}
MIXED_BLIMEY = {}

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
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, None,
            discretizer=fudd.QuartileDiscretizer)
    assert str(exin.value) == msg

    msg = ('This functionality requires the local model to be capable of '
           'outputting predictions via predict method.')
    model = fum.KNN()
    with pytest.raises(IncompatibleModelError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, InvalidModel,
            discretizer=fudd.QuartileDiscretizer)
    assert str(exin.value) == msg

    msg = ('The following indices are invalid for the input dataset: '
           '{}.'.format(np.array(['a'])))
    with pytest.raises(IndexError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, ['a'],
            discretizer=fudd.QuartileDiscretizer)
    assert str(exin.value) == msg

    msg = ('The categorical_indices parameter must be a Python list or None.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, 'a')
    assert str(exin.value) == msg

    msg = ('The augmentor object must inherit from abstract class '
           'fatf.utils.augmentation.Augmentation.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
                NUMERICAL_NP_ARRAY, InvalidModel,
                ftslm.SKLearnLinearModelExplainer, model, fum.KNN)
    assert str(exin.value) == msg

    msg = ('The discretizer object must be None or inherit from abstract class '
           'fatf.utils.discretization.Discretization.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
            discretizer=InvalidModel,)
    assert str(exin.value) == msg

    msg = ('The explainer object must inherit from abstract class fatf.utils.'
           'explainers.Explainer.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            InvalidModel, model, fum.KNN, discretizer=fudd.QuartileDiscretizer)
    assert str(exin.value) == msg


    msg = 'The class_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, class_names='a')
    assert str(exin.value) == msg

    msg = ('The class_name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, class_names=[0])
    assert str(exin.value) == msg

    msg = 'The feature_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, feature_names='a')
    assert str(exin.value) == msg
    
    msg = ('The length of feature_names must be equal to the number of '
           'features in the dataset.')
    with pytest.raises(ValueError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, feature_names=['a'])
    assert str(exin.value) == msg

    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN, feature_names=[0, 1, 2, 3])
    assert str(exin.value) == msg

    msg = ('discretize_first must be None or a boolean.')
    with pytest.raises(TypeError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
            discretize_first='a', discretizer=fudd.QuartileDiscretizer)
    assert str(exin.value) == msg

    msg = ('discretize_first is True but discretizer object is None. In order '
           'to discretize the sampled data prior to training a local model, '
           'please specify a discretizer object.')
    with pytest.raises(ValueError) as exin:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
            discretize_first=True)
    assert str(exin.value) == msg

    msg = ('discretize_first is False but discretizer has been specified. The '
           'discretizer will be ignored and the data will not be discretized '
           'prior to training a local model.')
    with pytest.warns(UserWarning) as warning:
        ftmb._input_is_valid(
            NUMERICAL_NP_ARRAY, fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
            discretize_first=False, discretizer=fudd.QuartileDiscretizer)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    # All ok
    assert ftmb._input_is_valid(
        NUMERICAL_NP_ARRAY, fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
        discretizer=fudd.QuartileDiscretizer)

    assert ftmb._input_is_valid(
        NUMERICAL_NP_ARRAY, fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
        discretizer=None)

    assert ftmb._input_is_valid(
        NUMERICAL_NP_ARRAY, fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer, model, fum.KNN,
        discretize_first=True, discretizer=fudd.QuartileDiscretizer)


class TestBlimey():
    """
    Tests :class:`fatf.transparency.models.blimey.Blimey`.
    """
    knn_numerical = fum.KNN(k=3)
    knn_numerical.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    knn_numerical_structured = fum.KNN(k=3)
    knn_numerical_structured.fit(NUMERICAL_STRUCT_ARRAY,
                                 NUMERICAL_NP_ARRAY_TARGET)

    knn_categorical = fum.KNN(k=3)
    knn_categorical.fit(CATEGORICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    knn_categorical_structured = fum.KNN(k=3)
    knn_categorical_structured.fit(CATEGORICAL_STRUCT_ARRAY,
                                   NUMERICAL_NP_ARRAY_TARGET)

    knn_mixed = fum.KNN(k=3)
    knn_mixed.fit(MIXED_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    class_names = ['Class A', 'Class B', 'Class C']
    feature_names = ['A', 'B', 'C', 'D']

    numerical_blimey = ftmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretizer=fudd.QuartileDiscretizer)

    numerical_blimey_cat = ftmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        categorical_indices=[0],
        discretizer=fudd.QuartileDiscretizer)

    numerical_blimey_no_discretization = ftmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretize_first=False)

    numerical_blimey_structured = ftmb.Blimey(
        dataset=NUMERICAL_STRUCT_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical_structured,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretizer=fudd.QuartileDiscretizer,
        categorical_indices=['a', 'b'])

    categorical_blimey = ftmb.Blimey(
        dataset=CATEGORICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_categorical,
        local_model=Ridge,
        discretizer=fudd.QuartileDiscretizer)

    categorical_blimey_structured = ftmb.Blimey(
        dataset=CATEGORICAL_STRUCT_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_categorical_structured,
        local_model=Ridge,
        discretizer=fudd.QuartileDiscretizer)

    mixed_blimey = ftmb.Blimey(
        dataset=MIXED_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_mixed,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretizer=fudd.QuartileDiscretizer)

    def test_init(self):
        """
        Tests :class:`fatf.transparency.models.blimey.Blimey` class init.
        """
        # Class inherits from explainer but is not tree or linear
        class ExplainerUnknown(fue.Explainer):
            def __init__(self): pass
            def feature_importance(self): pass

        # Infer discretize_first
        blimey_infer_discretize = ftmb.Blimey(
            dataset=NUMERICAL_NP_ARRAY,
            augmentor=fuda.NormalSampling,
            explainer=ftslm.SKLearnLinearModelExplainer,
            global_model=self.knn_numerical,
            local_model=Ridge,
            categorical_indices=[0],
            discretizer=fudd.QuartileDiscretizer)

        assert blimey_infer_discretize.discretize_first

        '''
        # With SKLearnDecisionTreeExplainer
        blimey_infer_discretize = ftmb.Blimey(
            dataset=NUMERICAL_NP_ARRAY,
            augmentor=fuda.NormalSampling,
            explainer=ftslm.SKLearnDecisionTreeExplainer,
            global_model=knn_numerical,
            local_model=Ridge,
            class_names=['Class A', 'Class B', 'Class C'],
            feature_names=['A', 'B', 'C', 'D'],
            categorical_indices=[0])

        assert blimey_infer_discretize.discretize_first is False
        '''

        msg = ('Unable to infer value of discretize_first as the explainer '
               'used is not one of the explainers defined in fatf.transparency.'
               'sklearn')
        with pytest.raises(ValueError) as exin:
            blimey_infer_discretize = ftmb.Blimey(
                dataset=NUMERICAL_NP_ARRAY,
                augmentor=fuda.NormalSampling,
                explainer=ExplainerUnknown,
                global_model=self.knn_numerical,
                local_model=Ridge,
                categorical_indices=[0],
                discretizer=fudd.QuartileDiscretizer)
        assert str(exin.value) == msg

        msg = ('discretize_first has been inferred to be True given the type '
               'of explainer, but a discretier object has not been given. As '
               'such, the data will not be discretized.')
        with pytest.warns(UserWarning) as warning:
            blimey_infer_discretize = ftmb.Blimey(
                dataset=NUMERICAL_NP_ARRAY,
                augmentor=fuda.NormalSampling,
                explainer=ftslm.SKLearnLinearModelExplainer,
                global_model=self.knn_numerical,
                local_model=Ridge,
                categorical_indices=[0])
        assert len(warning) == 1
        assert str(warning[0].message) == msg
        assert blimey_infer_discretize.discretize_first == False

        # numerical_blimey
        assert self.numerical_blimey.discretize_first == True
        assert self.numerical_blimey.is_structured == False
        assert self.numerical_blimey.categorical_indices == []
        assert self.numerical_blimey.numerical_indices == [0, 1, 2, 3]
        assert self.numerical_blimey.class_names == self.class_names
        assert self.numerical_blimey.feature_names == self.feature_names

        # numerical_blimey_cat
        assert self.numerical_blimey_cat.discretize_first == True
        assert self.numerical_blimey_cat.is_structured == False
        assert self.numerical_blimey_cat.categorical_indices == [0]
        assert self.numerical_blimey_cat.numerical_indices == [1, 2, 3]
        assert self.numerical_blimey_cat.class_names == self.class_names
        assert self.numerical_blimey_cat.feature_names == self.feature_names

        # numerical_blimey_no_discretization
        assert self.numerical_blimey_no_discretization.discretize_first == False
        assert self.numerical_blimey_no_discretization.is_structured == False
        assert self.numerical_blimey_no_discretization.categorical_indices == []
        assert (self.numerical_blimey_no_discretization.numerical_indices ==
                [0, 1, 2, 3])
        assert self.numerical_blimey_cat.class_names == self.class_names
        assert (self.numerical_blimey_no_discretization.feature_names ==
                self.feature_names)

        # numerical_blimey_structured
        assert self.numerical_blimey_structured.discretize_first == True
        assert self.numerical_blimey_structured.is_structured == True
        assert self.numerical_blimey_structured.categorical_indices == ['a', 'b']
        assert (self.numerical_blimey_structured.numerical_indices ==
                ['c', 'd'])
        assert self.numerical_blimey_structured.class_names == self.class_names
        assert (self.numerical_blimey_structured.feature_names ==
                self.feature_names)

        # categorical_blimey
        assert self.categorical_blimey.discretize_first == True
        assert self.categorical_blimey.is_structured == False
        assert self.categorical_blimey.categorical_indices == [0, 1, 2]
        assert self.categorical_blimey.numerical_indices == []
        assert self.categorical_blimey.class_names == ['class 0', 'class 1',
                                                       'class 2']
        assert (self.categorical_blimey.feature_names ==
                ['feature 0','feature 1', 'feature 2'])

        # categorical_blimey_structured
        assert self.categorical_blimey_structured.discretize_first == True
        assert self.categorical_blimey_structured.is_structured == True
        assert (self.categorical_blimey_structured.categorical_indices ==
                ['a', 'b', 'c'])
        assert self.categorical_blimey_structured.numerical_indices == []
        assert self.categorical_blimey_structured.class_names == ['class 0', 'class 1',
                                                       'class 2']
        assert (self.categorical_blimey.feature_names ==
                ['feature 0','feature 1', 'feature 2'])

        # mixed_blimey
        assert self.mixed_blimey.discretize_first == True
        assert self.mixed_blimey.is_structured == True
        assert self.mixed_blimey.categorical_indices == ['b', 'd']
        assert self.mixed_blimey.numerical_indices == ['a', 'c']
        assert self.mixed_blimey.class_names == self.class_names
        assert self.mixed_blimey.feature_names == self.feature_names

    def test_explain_instance_is_input_valid(self):
        """
        Tests :func:`fatf.transparency.models.blimey.Blimey._explain_instance
        _is_input_valid`.
        """
        assert True

    def test_explain_instance(self):
        """
        Tests :func:`fatf.transparency.models.blimey.Blimey.explain_instance`.
        """
        fatf.setup_random_seed()
        exp = self.numerical_blimey.explain_instance(NUMERICAL_NP_ARRAY[0])
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY)

        exp = self.numerical_blimey_cat.explain_instance(NUMERICAL_NP_ARRAY[0])
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_CAT)

        exp = self.numerical_blimey_no_discretization.explain_instance(
            NUMERICAL_NP_ARRAY[0])
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_NO_DISC)

        exp = self.numerical_blimey_structured.explain_instance(
            NUMERICAL_STRUCT_ARRAY[0])
        assert _is_explanation_equal(exp ,NUMERICAL_STRUCT_BLIMEY)
