"""
Tests surrogate_explainers classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Union, Tuple

import pytest

import numpy as np

from sklearn.linear_model import Ridge

import fatf

import fatf.transparency.sklearn.surrogate_explainers as ftsse
import fatf.utils.models as fum
import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretisation as fudd

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

Index = Union[int, str]


ONE_D_ARRAY = np.array([0, 4, 3, 0])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY = np.array([[0, 0, 0.08, 0.69], [1, 0, 0.03, 0.29],
                               [0, 1, 0.99, 0.82], [2, 1, 0.73, 0.48],
                               [1, 0, 0.36, 0.89], [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array([(0, 0, 0.08, 0.69), (1, 0, 0.03, 0.29),
                                   (0, 1, 0.99, 0.82), (2, 1, 0.73, 0.48),
                                   (1, 0, 0.36, 0.89), (0, 1, 0.07, 0.21)],
                                  dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'),
                                         ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([['a', 'b', 'c'], ['a', 'f', 'g'],
                                 ['b', 'c', 'c'], ['b', 'f', 'c'],
                                 ['a', 'f', 'c'], ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array([('a', 'b', 'c'), ('a', 'f', 'g'),
                                     ('b', 'c', 'c'), ('b', 'f', 'c'),
                                     ('a', 'f', 'c'), ('a', 'b', 'g')],
                                    dtype=[('a', 'U1'), ('b', 'U1'), ('c',
                                                                      'U1')])
MIXED_ARRAY = np.array([(0, 'a', 0.08, 'a'), (0, 'f', 0.03, 'bb'),
                        (1, 'c', 0.99, 'aa'), (1, 'a', 0.73, 'a'),
                        (0, 'c', 0.36, 'b'), (1, 'f', 0.07, 'bb')],
                       dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d',
                                                                    'U2')])

# Test LIME and bLIMEy on IRIS dataset
IRIS_DATASET = fatf_datasets.load_iris()


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


class NonProbaModel(object):
    """
    Tests for exceptions when model has ``predict`` method.
    """

    def __init__(self):
        """
        Initialises non-probabilistic.
        """
        pass

    def fit(self, X, y):
        """
        Fits non-probabilistic.
        """
        return X, y

    def predict(self, x):
        """
        Predicts non-probabilistic model.
        """
        pass


def _is_explanation_equal(dict1: Dict[str, Dict[Index, np.float64]],
                          dict2: Dict[str, Dict[Index, np.float64]],
                          tol: float = 1e-2) -> bool:
    """
    Tests if two explanations are equal within a tolerance.
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
                    if not np.isclose(feat_vals1, feat_vals2, tol):
                        equal = False
            else:
                equal = False
    else:
        equal = False
    return equal

def _is_bin_sampling_equal(dict1: Dict[str, Dict[Index, Tuple[float, ...]]],
                           dict2: Dict[str, Dict[Index, Tuple[float, ...]]],
                           tol: float = 1e-2) -> bool:
    """
    Tests if two bin sampling dictionarys are equal within a tolerance.
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
                    for (val1, val2) in zip(feat_vals1, feat_vals2):
                        if not np.isclose(val1, val2, tol):
                            equal = False
            else:
                equal = False
    else:
        equal = False
    return equal


def test_input_is_valid():
    """
    Tests :func:`fatf.transparency.sklearn.surrogate_explainers._is_input_valid`.
    """
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftsse._input_is_valid(ONE_D_ARRAY, None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            np.array([[0, None], [0, 8]]), None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('Non-probabilistic functionality requires the global model to be '
           'capable of outputting probabilities via predict method.')
    model = InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, False, None, None, None)
    assert str(exin.value) == msg

    msg = ('Probabilistic functionality requires the global model to be '
           'capable of outputting probabilities via predict_proba method.')
    model = NonProbaModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, None)
    assert str(exin.value) == msg

    msg = ('The following indices are invalid for the input dataset: '
           '{}.'.format(np.array(['a'])))
    model = fum.KNN(k=3)
    with pytest.raises(IndexError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, ['a'], None, None)
    assert str(exin.value) == msg

    msg = ('The categorical_indices parameter must be a Python list or None.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(NUMERICAL_NP_ARRAY, model, True, 'a', None, None)
    assert str(exin.value) == msg

    msg = 'The class_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, 'a', None)
    assert str(exin.value) == msg

    msg = ('The class_name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, [0], None)
    assert str(exin.value) == msg

    msg = 'The feature_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, 'a')
    assert str(exin.value) == msg

    msg = ('The length of feature_names must be equal to the number of '
           'features in the dataset.')
    with pytest.raises(ValueError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, ['a'])
    assert str(exin.value) == msg

    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, [0, 1, 2, 3])
    assert str(exin.value) == msg

    # All ok
    assert ftsse._input_is_valid(NUMERICAL_NP_ARRAY, model, True, [0],
                                 ['class 1'], ['1', '2', '3', '4'])


class TestSurrogateExplainer():
    """
    Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
    SurrogateExplainer` abstract class.
    """

    class BrokenSurrogateExplainer(ftsse.SurrogateExplainer):
        """
        A broken surrogate explainer implementation.

        This class does not have a ``explain_instance`` method.
        """

        def __init__(self, dataset, global_model, probabilistic,
                     categorical_indices, class_names, feature_names):
            """
            Dummy init method.
            """
            super().__init__(dataset, global_model, probabilistic,
                             categorical_indices, class_names, feature_names)

    class BaseSurrogateExplainer(ftsse.SurrogateExplainer):
        """
        A dummy surrogate explainer implementation.
        """

        def __init__(self, dataset, global_model, probabilistic=True,
                     categorical_indices=None, class_names=None,
                     feature_names=None):
            """
            Dummy init method.
            """
            super().__init__(
                dataset, global_model, probabilistic, categorical_indices,
                class_names, feature_names)

        def explain_instance(self, data_row):
            """
            Dummy explain_insatnce method.
            """
            self._explain_instance_input_is_valid(data_row)
            return {'0':{'0': 1}}

    numerical_np_array_classifier = fum.KNN(k=3)
    numerical_np_array_classifier.fit(NUMERICAL_NP_ARRAY,
                                      NUMERICAL_NP_ARRAY_TARGET)

    numerical_struct_array_classifier = fum.KNN(k=3)
    numerical_struct_array_classifier.fit(NUMERICAL_STRUCT_ARRAY,
                                          NUMERICAL_NP_ARRAY_TARGET)

    categorical_np_array_classifier = fum.KNN(k=3)
    categorical_np_array_classifier.fit(CATEGORICAL_NP_ARRAY,
                                        NUMERICAL_NP_ARRAY_TARGET)

    categorical_struct_array_classifier = fum.KNN(k=3)
    categorical_struct_array_classifier.fit(CATEGORICAL_STRUCT_ARRAY,
                                            NUMERICAL_NP_ARRAY_TARGET)

    mixed_classifier = fum.KNN(k=3)
    mixed_classifier.fit(MIXED_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    numerical_dummy_surrogate = BaseSurrogateExplainer(
        NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier,
        probabilistic=True,
        categorical_indices=[0],
        class_names=None,
        feature_names=['1', '2', '3', '4'])

    numerical_struct_dummy_surrogate = BaseSurrogateExplainer(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_array_classifier,
        probabilistic=False,
        categorical_indices=['a', 'b'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=None)

    categorical_np_dummy_surrogate = BaseSurrogateExplainer(
        CATEGORICAL_NP_ARRAY,
        categorical_np_array_classifier,
        categorical_indices=[0, 1, 2])

    categorical_struct_dummy_surrogate = BaseSurrogateExplainer(
        CATEGORICAL_STRUCT_ARRAY,
        categorical_struct_array_classifier,
        categorical_indices=['a', 'b', 'c'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=['1', '2', '3'])

    mixed_dummy_surrogate = BaseSurrogateExplainer(
        MIXED_ARRAY,
        mixed_classifier,
        categorical_indices=['b', 'd'],
        feature_names=['num1', 'str1', 'num2', 'str2'])

    def test_surrogate_explainer_init(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_edplainers.\
        SurrogateExplainer.__init__` initialiser.
        """
        abstract_method_error = ("Can't instantiate abstract class "
                                 "{} with abstract methods explain_instance")

        user_warning = (
            'Some of the string-based columns in the input dataset were not '
            'selected as categorical features via the categorical_indices '
            'parameter. String-based columns cannot be treated as numerical '
            'features, therefore they will be also treated as categorical '
            'features (in addition to the ones selected with the '
            'categorical_indices parameter).')

        # Don't define explain_instance method.
        with pytest.raises(TypeError) as exin:
            self.BrokenSurrogateExplainer(NUMERICAL_NP_ARRAY,
                                          self.numerical_np_array_classifier)
        msg = abstract_method_error.format('BrokenSurrogateExplainer')
        assert str(exin.value) == msg
        #
        with pytest.raises(TypeError) as exin:
            ftsse.SurrogateExplainer(NUMERICAL_NP_ARRAY,
                                     self.numerical_np_array_classifier)
        assert str(exin.value) == abstract_method_error.format(
            'SurrogateExplainer')

        # Warning for handling of categorical indices.
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                CATEGORICAL_NP_ARRAY, self.categorical_np_array_classifier,
                True, [0])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              [0, 1, 2])
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                CATEGORICAL_STRUCT_ARRAY,
                self.categorical_struct_array_classifier,
                True,
                ['a'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              np.array(['a', 'b', 'c']))
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                MIXED_ARRAY, self.mixed_classifier, True, ['b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              ['b', 'd'])

        # Validate internal variables
        assert np.array_equal(self.numerical_dummy_surrogate.dataset,
                              NUMERICAL_NP_ARRAY)
        assert not self.numerical_dummy_surrogate.is_structured
        assert self.numerical_dummy_surrogate.categorical_indices == [0]
        assert self.numerical_dummy_surrogate.numerical_indices == [1, 2, 3]
        assert self.numerical_dummy_surrogate.n_classes == 3
        assert list(self.numerical_dummy_surrogate.indices) == [0, 1, 2, 3]
        assert self.numerical_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.numerical_dummy_surrogate.feature_names == \
            ['1', '2', '3', '4']
        assert self.numerical_dummy_surrogate.prediction_function == \
            self.numerical_np_array_classifier.predict_proba

        assert np.array_equal(self.numerical_struct_dummy_surrogate.dataset,
                              NUMERICAL_STRUCT_ARRAY)
        assert self.numerical_struct_dummy_surrogate.is_structured
        assert self.numerical_struct_dummy_surrogate.categorical_indices == \
            ['a', 'b']
        assert self.numerical_struct_dummy_surrogate.numerical_indices == \
            ['c', 'd']
        assert self.numerical_struct_dummy_surrogate.n_classes == 3
        assert list(self.numerical_struct_dummy_surrogate.indices) == \
            ['a', 'b', 'c', 'd']
        assert self.numerical_struct_dummy_surrogate.class_names == \
            ['class1', 'class2', 'class3']
        assert self.numerical_struct_dummy_surrogate.feature_names == \
            ['feature 0', 'feature 1', 'feature 2', 'feature 3']
        assert self.numerical_struct_dummy_surrogate.prediction_function == \
            self.numerical_struct_array_classifier.predict

        assert np.array_equal(self.categorical_np_dummy_surrogate.dataset,
                              CATEGORICAL_NP_ARRAY)
        assert not self.categorical_np_dummy_surrogate.is_structured
        assert self.categorical_np_dummy_surrogate.categorical_indices == \
            [0, 1, 2]
        assert self.categorical_np_dummy_surrogate.numerical_indices == []
        assert self.categorical_np_dummy_surrogate.n_classes == 3
        assert list(self.categorical_np_dummy_surrogate.indices) == \
            [0, 1, 2]
        assert self.categorical_np_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.categorical_np_dummy_surrogate.feature_names == \
            ['feature 0', 'feature 1', 'feature 2']
        assert self.categorical_np_dummy_surrogate.prediction_function == \
            self.categorical_np_array_classifier.predict_proba

        assert np.array_equal(self.categorical_struct_dummy_surrogate.dataset,
                              CATEGORICAL_STRUCT_ARRAY)
        assert self.categorical_struct_dummy_surrogate.is_structured
        assert self.categorical_struct_dummy_surrogate.categorical_indices == \
            ['a', 'b', 'c']
        assert self.categorical_struct_dummy_surrogate.numerical_indices == []
        assert self.categorical_struct_dummy_surrogate.n_classes == 3
        assert list(self.categorical_struct_dummy_surrogate.indices) == \
            ['a', 'b', 'c']
        assert self.categorical_struct_dummy_surrogate.class_names == \
            ['class1', 'class2', 'class3']
        assert self.categorical_struct_dummy_surrogate.feature_names == \
            ['1', '2', '3']
        assert self.categorical_struct_dummy_surrogate.prediction_function == \
            self.categorical_struct_array_classifier.predict_proba

        assert np.array_equal(self.mixed_dummy_surrogate.dataset,
                              MIXED_ARRAY)
        assert self.mixed_dummy_surrogate.is_structured
        assert self.mixed_dummy_surrogate.categorical_indices == ['b', 'd']
        assert self.mixed_dummy_surrogate.numerical_indices == ['a', 'c']
        assert self.mixed_dummy_surrogate.n_classes == 3
        assert list(self.mixed_dummy_surrogate.indices) == \
            ['a', 'b', 'c', 'd']
        assert self.mixed_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.mixed_dummy_surrogate.feature_names == \
            ['num1', 'str1', 'num2', 'str2']
        assert self.mixed_dummy_surrogate.prediction_function == \
            self.mixed_classifier.predict_proba

    def test_explain_instance_validation(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_explainers.\
        SurrogateExplainer._explain_instance_is_valid` method.
        """
        incorrect_shape_data_row = ('The data_row must either be a '
                                    '1-dimensional numpy array or numpy void '
                                    'object for structured rows.')
        type_error_data_row = ('The dtype of the data_row is different to the '
                               'dtype of the data array used to initialise '
                               'this class.')
        incorrect_shape_features = ('The data_row must contain the same '
                                    'number of features as the dataset used '
                                    'to initialise this class.')

        # data_row shape
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(NUMERICAL_NP_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                NUMERICAL_STRUCT_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row

        # data_row type
        with pytest.raises(TypeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(np.array(['a', 'b', 'c', 'd']))
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                MIXED_ARRAY[0])
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.categorical_np_dummy_surrogate.explain_instance(
                np.array([0.1]))
        assert str(exin.value) == type_error_data_row
        # Structured too short
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                MIXED_ARRAY[['a', 'b']][0])
        assert str(exin.value) == type_error_data_row

        # data_row features number
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(
                np.array([0.1, 1, 2]))
        assert str(exin.value) == incorrect_shape_features
        with pytest.raises(IncorrectShapeError) as exin:
            self.categorical_np_dummy_surrogate.explain_instance(
                np.array(['a', 'b']))
        assert str(exin.value) == incorrect_shape_features


class TestTabularBlimeyTree():
    """
    Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
    TabularBlimeyTree` abstract class.
    """
    numerical_np_array_classifier = fum.KNN(k=3)
    numerical_np_array_classifier.fit(NUMERICAL_NP_ARRAY,
                                      NUMERICAL_NP_ARRAY_TARGET)

    numerical_struct_array_classifier = fum.KNN(k=3)
    numerical_struct_array_classifier.fit(NUMERICAL_STRUCT_ARRAY,
                                          NUMERICAL_NP_ARRAY_TARGET)

    categorical_array_classifier = fum.KNN(k=3)
    categorical_array_classifier.fit(CATEGORICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET)

    iris_classifier = fum.KNN(k=3)
    iris_classifier.fit(IRIS_DATASET['data'], IRIS_DATASET['target'])

    numerical_np_tabular_lime = ftsse.TabularBlimeyTree(
        NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier)

    numerical_np_cat_tabular_lime= ftsse.TabularBlimeyTree(
        NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier,
        categorical_indices=[0, 1])

    iris_blimey = ftsse.TabularBlimeyTree(
        IRIS_DATASET['data'],
        iris_classifier,
        class_names=list(IRIS_DATASET['target_names']),
        feature_names=list(IRIS_DATASET['feature_names']))

    def test_tabular_blimey_tree_init(self):
        """
        Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
        TabularBlimeyTree` class init.
        """
        # Test class inheritance
        assert (self.numerical_np_tabular_blimey.__class__.__bases__[0].__name__
                == 'SurrogateExplainer')

        string_array_error = ('TabularBlimeyTree does not support string '
                              'dtype as it uses sci-kit learn '
                              'implementation of decision trees.')
        structured_array_error = ('TabularBlimeyTree does not support '
                                  'structured arrays as it uses sci-kit learn '
                                  'implementation of decision trees.')

        with pytest.raises(TypeError) as exin:
            ftsse.TabularBlimeyTree(NUMERICAL_STRUCT_ARRAY,
                                    self.numerical_struct_array_classifier)
        assert str(exin.value) == structured_array_error

        with pytest.raises(TypeError) as exin:
            ftsse.TabularBlimeyTree(CATEGORICAL_NP_ARRAY,
                                     self.categorical_array_classifier)
        assert str(exin.value) == string_array_error

        # Assert indices
        assert self.numerical_np_tabular_blimey.numerical_indices == \
            [0, 1, 2, 3]
        assert self.numerical_np_tabular_blimey.categorical_indices == []
        assert isinstance(self.numerical_np_tabular_blimey.augmentor,
                          fuda.Mixup)
        #
        assert self.numerical_np_cat_tabular_blimey.numerical_indices == \
            [2, 3]
        assert self.numerical_np_cat_tabular_blimey.categorical_indices == \
            [0, 1]
        assert isinstance(self.numerical_np_tabular_blimey.augmentor,
                          fuda.Mixup)

    def test_tabular_blimey_tree_explain_instance_input_is_valid(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_explainers.\
        TabularBlimeyTree._explain_instance_input_is_valid`.
        """
        samples_number_type = ('samples_number must be an integer.')
        samples_number_value = ('samples_number must be a positive integer '
                                'larger than 0.')
        maximum_depth_type = ('maximum_depth must be an integer.')
        maximum_depth_value = ('maximum_depth must be a positive integer '
                                'larger than 0.')

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 'a', 3)
        assert str(exin.value) == samples_number_type
        #
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], -1, 3)
        assert str(exin.value) == samples_number_value
        #
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, 'a')
        assert str(exin.value) == maximum_depth_type
        #
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, -1)
        assert str(exin.value) == maximum_depth_value
        # All good
        assert self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
            NUMERICAL_NP_ARRAY[0], 10, 3)
        assert self.numerical_np_cat_tabular_blimey._explain_instance_input_is_valid(
            NUMERICAL_NP_ARRAY[0], 10, 3)

    def test_tabular_blimey_tree_explain_instance(self):
        fatf.setup_random_seed()

        numerical_np_explanation = {
            'class 0': {
                'feature 0': 0.074,
                'feature 1': 0.797,
                'feature 2': 0.130,
                'feature 3': 0.0},
            'class 1': {
                'feature 0': 0.0,
                'feature 1': 0.149,
                'feature 2': 0.851,
                'feature 3': 0.0},
            'class 2': {
                'feature 0': 0.052,
                'feature 1': 0.408,
                'feature 2': 0.540,
                'feature 3': 0.0}}

        numerical_np_cat_explanation = {
            'class 0': {
                'feature 0': 0.071,
                'feature 1': 0.460,
                'feature 2': 0.197,
                'feature 3': 0.272},
            'class 1': {
                'feature 0': 0.0,
                'feature 1': 0.534,
                'feature 2': 0.466,
                'feature 3': 0.0},
            'class 2': {
                'feature 0': 0.166,
                'feature 1': 0.151,
                'feature 2': 0.0,
                'feature 3': 0.683}}

        iris_explanation = {
            'setosa': {
                'petal length (cm)': 0.0044,
                'petal width (cm)': 0.996,
                'sepal length (cm)': 0.0,
                'sepal width (cm)': 0.0},
            'versicolor': {
                'petal length (cm)': 0.0,
                'petal width (cm)': 1.0,
                'sepal length (cm)': 0.0,
                'sepal width (cm)': 0.0},
            'virginica': {
                'petal length (cm)': 0.0,
                'petal width (cm)': 0.093,
                'sepal length (cm)': 0.817,
                'sepal width (cm)': 0.089}}

        explanation = self.numerical_np_tabular_blimey.explain_instance(
            NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=3,
            random_state=42)
        assert _is_explanation_equal(numerical_np_explanation, explanation)

        explanation = self.numerical_np_cat_tabular_blimey.explain_instance(
            NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=3,
            random_state=42)
        assert _is_explanation_equal(numerical_np_cat_explanation, explanation)

        # Test IRIS
        explanation = self.iris_blimey.explain_instance(
            IRIS_DATASET['data'][0],
            samples_number=50,
            maximum_depth=3,
            random_state=42)
        assert _is_explanation_equal(iris_explanation, explanation)


class TestTabularLIME():
    """
    Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
    TabularLIME` class.
    """

    numerical_np_array_classifier = fum.KNN(k=3)
    numerical_np_array_classifier.fit(NUMERICAL_NP_ARRAY,
                                      NUMERICAL_NP_ARRAY_TARGET)

    numerical_struct_array_classifier = fum.KNN(k=3)
    numerical_struct_array_classifier.fit(NUMERICAL_STRUCT_ARRAY,
                                          NUMERICAL_NP_ARRAY_TARGET)

    categorical_array_classifier = fum.KNN(k=3)
    categorical_array_classifier.fit(CATEGORICAL_NP_ARRAY,
                                     NUMERICAL_NP_ARRAY_TARGET)

    iris_classifier = fum.KNN(k=3)
    iris_classifier.fit(IRIS_DATASET['data'], IRIS_DATASET['target'])

    numerical_np_tabular_lime = ftsse.TabularLIME(
        NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier)

    numerical_struct_cat_tabular_lime = ftsse.TabularLIME(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_array_classifier,
        categorical_indices=['a', 'b'])

    categorical_np_lime = ftsse.TabularLIME(
        CATEGORICAL_NP_ARRAY,
        categorical_array_classifier,
        categorical_indices=[0, 1, 2])

    iris_lime = ftsse.TabularLIME(
        IRIS_DATASET['data'],
        iris_classifier,
        class_names=list(IRIS_DATASET['target_names']),
        feature_names=list(IRIS_DATASET['feature_names']))

    def test_tabular_LIME_init(self):
        """
        Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
        TabularLIME` class init.
        """
        numerical_bin_sampling_values = {
            0: { # Index 0
                0: (0.0, 0.0, 0.0, 0.0),
                2: (1.0, 1.0, 1.0, 0.0),
                3: (2.0, 2.0, 2.0, 0.0)},
            1: { # Index 1
                0: (0.0, 0.0, 0.0, 0.0),
                2: (1.0, 1.0, 1.0, 0.0)},
            2: { # Index 2
                0: (0.03, 0.07, 0.05, 0.020),
                1: (0.08, 0.08, 0.08, 0.0),
                2: (0.36, 0.36, 0.36, 0.0),
                3: (0.73, 0.99, 0.86, 0.13)},
            3: { # Index 3
                0: (0.21, 0.29, 0.25, 0.04),
                1: (0.48, 0.48, 0.48, 0.0),
                2: (0.69, 0.69, 0.69, 0.0),
                3: (0.82, 0.89, 0.855, 0.035)}}
        numerical_struct_sampling_values = {
            'c': numerical_bin_sampling_values[2],
            'd': numerical_bin_sampling_values[3]}
        iris_lime_sampling_values = {
            0: {
                0: (4.3, 5.1, 4.856, 0.228),
                1: (5.2, 5.8, 5.559, 0.185),
                2: (5.9, 6.4, 6.189, 0.163),
                3: (6.5, 7.9, 6.971, 0.412)},
            1: {
                0: (2.0, 2.8, 2.585, 0.208),
                1: (2.9, 3.0, 2.972, 0.045),
                2: (3.1, 3.3, 3.183, 0.073),
                3: (3.4, 4.4, 3.638, 0.252)},
            2: {
                0: (1.0, 1.6, 1.420, 0.134),
                1: (1.7, 4.3, 3.474, 0.890),
                2: (4.4, 5.1, 4.766, 0.243),
                3: (5.2, 6.9, 5.826, 0.437)},
            3: {
                0: (0.1, 0.3, 0.205, 0.054),
                1: (0.4, 1.3, 1.003, 0.342),
                2: (1.4, 1.8, 1.595, 0.157),
                3: (1.9, 2.5, 2.171, 0.187)}}

        # Assert indices
        assert self.numerical_np_tabular_lime.numerical_indices == \
            [0, 1, 2, 3]
        assert self.numerical_np_tabular_lime.categorical_indices == []
        assert isinstance(self.numerical_np_tabular_lime.augmentor,
                          fuda.NormalSampling)
        assert isinstance(self.numerical_np_tabular_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert _is_bin_sampling_equal(self.numerical_np_tabular_lime.\
            bin_sampling_values, numerical_bin_sampling_values)
        #
        assert self.numerical_struct_cat_tabular_lime.numerical_indices == \
            ['c', 'd']
        assert self.numerical_struct_cat_tabular_lime.categorical_indices == \
            ['a', 'b']
        assert isinstance(self.numerical_struct_cat_tabular_lime.augmentor,
                          fuda.NormalSampling)
        assert isinstance(self.numerical_struct_cat_tabular_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert _is_bin_sampling_equal(self.numerical_struct_cat_tabular_lime.\
            bin_sampling_values, numerical_struct_sampling_values)
        #
        assert self.categorical_np_lime.numerical_indices == \
            []
        assert self.categorical_np_lime.categorical_indices == \
            [0, 1, 2]
        assert isinstance(self.categorical_np_lime.augmentor,
                          fuda.NormalSampling)
        assert isinstance(self.categorical_np_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert self.categorical_np_lime.bin_sampling_values == {}
        #
        assert self.iris_lime.numerical_indices == \
            [0, 1, 2, 3]
        assert self.iris_lime.categorical_indices == []
        assert isinstance(self.iris_lime.augmentor,
                          fuda.NormalSampling)
        assert isinstance(self.iris_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert _is_bin_sampling_equal(self.iris_lime.bin_sampling_values,
                iris_lime_sampling_values)

    def test_tabular_LIME_explain_instance_input_is_valid(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_explainers.\
        TabularLIME._explain_instance_input_is_valid`.
        """
        samples_number_type = ('samples_number must be an integer.')
        samples_number_value = ('samples_number must be a positive integer '
                                'larger than 0.')
        features_number_type = ('features_number must be an integer.')
        features_number_value = ('features_number must be a positive integer '
                                'larger than 0.')
        kernel_width_type = ('kernel_width must be None or a float.')
        kernel_width_value = ('kernel_width must be None or a positive float '
                                'larger than 0.')
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 'a', 3, 1.0)
        assert str(exin.value) == samples_number_type
        #
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], -1, 3, 1.0)
        assert str(exin.value) == samples_number_value
        #
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, 'a', 1.0)
        assert str(exin.value) == features_number_type
        #
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, -1, 1.0)
        assert str(exin.value) == features_number_value
        #
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, 2, 'a')
        assert str(exin.value) == kernel_width_type
        #
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                NUMERICAL_NP_ARRAY[0], 1, 2, -1.0)
        assert str(exin.value) == kernel_width_value
        # All good
        assert self.numerical_np_tabular_lime._explain_instance_input_is_valid(
            NUMERICAL_NP_ARRAY[0], 10, 3, 1.0)
        assert self.numerical_np_tabular_lime._explain_instance_input_is_valid(
            NUMERICAL_NP_ARRAY[0], 10, 3, 1.0)

    def test_tabular_LIME_explain_instance(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_explainers.\
        TabularLIME.explain_instance`.
        """
        fatf.setup_random_seed()
        numerical_np_explanation = {}

        explanation = self.numerical_np_tabular_lime.explain_instance(
            NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            features_number=2,
            kernel_width=None,
            random_state=42)
        assert _is_explanation_equal(numerical_np_explanation, explanation)
