"""
Tests scikit-learn transparency tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import sklearn
    import sklearn.tree
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of scikit-learn transparency tools '
        '-- scikit-learn is not installed.',
        allow_module_level=True)

import numpy as np

import fatf.transparency.sklearn.tools as ftst

IOBJ = object()
BOBJ = sklearn.base.BaseEstimator()
TOBJ = sklearn.tree.DecisionTreeClassifier()


def test_is_sklearn_model():
    """
    Tests :func:`fatf.transparency.sklearn.tools.is_sklearn_model` function.
    """
    assert not ftst.is_sklearn_model(IOBJ)
    assert not ftst.is_sklearn_model(object)
    assert ftst.is_sklearn_model(BOBJ)
    assert ftst.is_sklearn_model(sklearn.base.BaseEstimator)
    assert ftst.is_sklearn_model(TOBJ)
    assert ftst.is_sklearn_model(sklearn.tree.DecisionTreeClassifier)


def test_is_sklearn_model_instance():
    """
    Tests :func:`fatf.transparency.sklearn.tools.is_sklearn_model_instance`.
    """
    assert not ftst.is_sklearn_model_instance(IOBJ)
    assert not ftst.is_sklearn_model_instance(object)
    assert ftst.is_sklearn_model_instance(BOBJ)
    assert not ftst.is_sklearn_model_instance(sklearn.base.BaseEstimator)
    assert ftst.is_sklearn_model_instance(TOBJ)
    assert not ftst.is_sklearn_model_instance(
        sklearn.tree.DecisionTreeClassifier)


def test_validate_input():
    """
    Tests :func:`fatf.transparency.sklearn.tools._validate_input` function.
    """
    type_error_clf = ('The model has to be a scikit-learn classifier, i.e., '
                      'it has to inherit from sklearn.base.BaseEstimator.')
    type_error_fno = ('The feature_names parameter has to be either None or a '
                      'Python list.')
    type_error_fni = ('All elements of the feature_names list have to be '
                      'strings.')
    value_error_fn = 'The feature names list cannot be empty.'
    type_error_cno = ('The class_names parameter has to be either None or a '
                      'Python list.')
    type_error_cni = 'All elements of the class_names list have to be strings.'
    value_error_cn = 'The class names list cannot be empty.'

    with pytest.raises(TypeError) as exin:
        ftst._validate_input(IOBJ, None, None)
    assert str(exin.value) == type_error_clf
    with pytest.raises(TypeError) as exin:
        ftst._validate_input(sklearn.tree.DecisionTreeClassifier, None, None)
    assert str(exin.value) == type_error_clf

    with pytest.raises(TypeError) as exin:
        ftst._validate_input(TOBJ, 'list', None)
    assert str(exin.value) == type_error_fno
    with pytest.raises(ValueError) as exin:
        ftst._validate_input(TOBJ, [], None)
    assert str(exin.value) == value_error_fn
    with pytest.raises(TypeError) as exin:
        ftst._validate_input(TOBJ, ['a', 'b', 3, 'd'], None)
    assert str(exin.value) == type_error_fni

    with pytest.raises(TypeError) as exin:
        ftst._validate_input(BOBJ, ['a', 'b', 'c', 'd'], 'list')
    assert str(exin.value) == type_error_cno
    with pytest.raises(ValueError) as exin:
        ftst._validate_input(BOBJ, None, [])
    assert str(exin.value) == value_error_cn
    with pytest.raises(TypeError) as exin:
        ftst._validate_input(BOBJ, ['a', 'b', 'c', 'd'], ['a', 3, 'c'])
    assert str(exin.value) == type_error_cni

    assert ftst._validate_input(BOBJ, None, None)
    assert ftst._validate_input(BOBJ, ['a', 'b'], None)
    assert ftst._validate_input(TOBJ, None, ['a', 'b'])
    assert ftst._validate_input(TOBJ, ['a', 'b'], ['a', 'b'])


class TestSKLearnExplainer(object):
    """
    Tests the :class:`fatf.transparency.sklearn.tools.SKLearnExplainer` class.
    """

    class SKLearnExplainer_validate_e(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return None

        def _is_classifier(self):
            # Will not be executed since this class is broken
            return True  # pragma: no cover

        def _get_features_number(self):
            # Will not be executed since this class is broken
            return None  # pragma: no cover

        def _get_classes_array(self):
            # Will not be executed since this class is broken
            return None  # pragma: no cover

    class SKLearnExplainer_classifier_e(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return True

        def _is_classifier(self):
            return None

        def _get_features_number(self):
            # Will not be executed since this class is broken
            return None  # pragma: no cover

        def _get_classes_array(self):
            # Will not be executed since this class is broken
            return None  # pragma: no cover

    class SKLearnExplainer_valid(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return True

        def _is_classifier(self):
            return True

        def _get_features_number(self):
            return None

        def _get_classes_array(self):
            return None

    class SKLearnExplainer_full(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return True

        def _is_classifier(self):
            return True

        def _get_features_number(self):
            return 3

        def _get_classes_array(self):
            return np.array(['x', 'y'])

    class SKLearnExplainer_invalid_reg(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return True

        def _is_classifier(self):
            return False

        def _get_features_number(self):
            return None

        def _get_classes_array(self):
            return np.array(['x', 'y'])

    class SKLearnExplainer_valid_reg(ftst.SKLearnExplainer):
        def _validate_kind_fitted(self):
            return True

        def _is_classifier(self):
            return False

        def _get_features_number(self):
            return None

        def _get_classes_array(self):
            return None

    def test_class_init(self, caplog):
        """
        Tests different ways of initialising ``SKLearnExplainer`` instances.
        """
        type_error_abstract = ("Can't instantiate abstract class "
                               'SKLearnExplainer with abstract methods '
                               '_get_classes_array, _get_features_number, '
                               '_is_classifier, _validate_kind_fitted')
        assertion_kind_fit = 'Unfitted or wrong type model.'
        assertion_classifier = 'Has to be boolean.'
        assertion_reg = ("Regressor's class_names and classes_array must both "
                         'be None.')

        value_error_fn = ('The length of the feature_names list is different '
                          'than the number of features extracted from the '
                          'classifier.')
        value_error_cn = ('The length of the class_names list is different '
                          'than the length of the classes array extracted '
                          'from the classifier.')

        log_info_fn = ('Generating missing feature names from the number of '
                       'features using "feature %d" pattern.')
        log_info_cn = ('Generating missing class names from the array of '
                       'classes output by the classifier using "class %s" '
                       'pattern.')

        user_warning_fn = ('Cannot validate the length of feature names list '
                           'since the _get_features_number method returned '
                           'None.')
        user_warning_cn = ('Cannot validate the length of class names list '
                           'since the _get_classes_array method returned '
                           'None.')

        with pytest.raises(TypeError) as exin:
            ftst.SKLearnExplainer()
        assert str(exin.value) == type_error_abstract

        # Test the enforced return type of _validate_kind_fitted
        with pytest.raises(AssertionError) as exin:
            ske = self.SKLearnExplainer_validate_e(TOBJ)
        assert str(exin.value) == assertion_kind_fit
        # Test the enforced return type of _is_classifier
        with pytest.raises(AssertionError) as exin:
            ske = self.SKLearnExplainer_classifier_e(TOBJ)
        assert str(exin.value) == assertion_classifier

        # Test
        ske = self.SKLearnExplainer_valid(TOBJ)
        assert ske.clf == TOBJ
        assert ske.feature_names is None
        assert ske.class_names is None
        assert ske.is_classifier is True
        assert ske.features_number is None
        assert ske.classes_array is None

        # Test warnings
        f_names = ['f', 'names']
        c_names = ['c', 'names']
        with pytest.warns(UserWarning) as w:
            ske = self.SKLearnExplainer_valid(TOBJ, f_names, c_names)
        assert len(w) == 2
        assert str(w[0].message) == user_warning_fn
        assert str(w[1].message) == user_warning_cn
        #
        assert ske.clf == TOBJ
        assert ske.feature_names == f_names
        assert ske.class_names == c_names
        assert ske.is_classifier is True
        assert ske.features_number is None
        assert ske.classes_array is None

        # Test
        f_names = ['a', 'b', 'c']
        c_names = ['42', '24']
        with pytest.raises(ValueError) as exin:
            self.SKLearnExplainer_full(TOBJ, ['a'], ['1'])
        assert str(exin.value) == value_error_fn
        with pytest.raises(ValueError) as exin:
            self.SKLearnExplainer_full(TOBJ, f_names, ['42'])
        assert str(exin.value) == value_error_cn

        ske = self.SKLearnExplainer_full(TOBJ, f_names, c_names)
        assert ske.clf == TOBJ
        assert ske.feature_names == f_names
        assert ske.class_names == c_names
        assert ske.is_classifier is True
        assert ske.features_number == 3
        assert np.array_equal(ske.classes_array, ['x', 'y'])

        # Test logging
        assert len(caplog.records) == 0
        ske = self.SKLearnExplainer_full(TOBJ)
        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage() == log_info_fn
        assert caplog.records[1].levelname == 'INFO'
        assert caplog.records[1].getMessage() == log_info_cn
        #
        assert ske.clf == TOBJ
        assert ske.feature_names == ['feature 0', 'feature 1', 'feature 2']
        assert ske.class_names == ['class x', 'class y']
        assert ske.is_classifier is True
        assert ske.features_number == 3
        assert np.array_equal(ske.classes_array, ['x', 'y'])

        # Regression with class names instead of None
        with pytest.raises(AssertionError) as exin:
            self.SKLearnExplainer_invalid_reg(TOBJ)
        assert str(exin.value) == assertion_reg

        with pytest.warns(UserWarning) as w:
            ske = self.SKLearnExplainer_valid_reg(TOBJ, feature_names=f_names)
        assert len(w) == 1
        assert str(w[0].message) == user_warning_fn
        assert ske.feature_names == f_names
        assert ske.class_names is None
        assert ske.is_classifier is False
        assert ske.features_number is None
        assert ske.classes_array is None

        with pytest.raises(AssertionError) as exin:
            self.SKLearnExplainer_valid_reg(TOBJ, class_names=['a', 'b'])
        assert str(exin.value) == assertion_reg

    def test_map_class(self):
        """
        Tests ``map_class`` method.

        Tests
        :func:`fatf.transparency.sklearn.tools.SKLearnExplainer.map_class`
        method.
        """
        runtime_error_reg = ('This functionality is not available for '
                             'regressors.')
        type_error_cls = ('The clf_class parameter must either be integer or '
                          'string.')
        runtime_error_none = ('class_names and/or classes_array have not been '
                              'initialised; mapping is not supported.')
        value_error_cls = ('Given class value is invalid for this classifier. '
                           'The following values are possible: {}.')

        ske = self.SKLearnExplainer_valid_reg(TOBJ)
        with pytest.raises(RuntimeError) as exin:
            ske.map_class('a')
        assert str(exin.value) == runtime_error_reg

        ske = self.SKLearnExplainer_valid(TOBJ)
        with pytest.raises(TypeError) as exin:
            ske.map_class(None)
        assert str(exin.value) == type_error_cls
        with pytest.raises(RuntimeError) as exin:
            ske.map_class('None')
        assert str(exin.value) == runtime_error_none

        ske = self.SKLearnExplainer_full(TOBJ, class_names=['first', 'second'])
        with pytest.raises(ValueError) as exin:
            ske.map_class('None')
        assert str(exin.value) == value_error_cls.format(['x', 'y'])

        assert ske.map_class('y') == 'second'
