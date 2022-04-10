"""
Tests surrogate explainers.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <K.Sokol@bristol.ac.uk>
# License: new BSD

import pytest

try:
    import sklearn
except ImportError:  # pragma: no cover
    _missing_sklearn_msg = (
        'The TabularBlimeyLime and TabularBlimeyTree surrogate explainers '
        'require scikit-learn to be installed. Since scikit-learn is missing, '
        'this functionality will be disabled.')
    with pytest.warns(UserWarning) as warning:
        import fatf.transparency.predictions.surrogate_explainers as ftps
    assert len(warning) > 0
    assert str(warning[-1].message) == _missing_sklearn_msg

    SKLEARN_MISSING = True
else:
    del sklearn
    import fatf.transparency.predictions.surrogate_explainers as ftps
    SKLEARN_MISSING = False

import importlib
import sys

import numpy as np

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

import fatf

import fatf.utils.array.tools as fuat
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.data.discretisation as fudd
import fatf.utils.models as fum
import fatf.utils.testing.transparency as futt
import fatf.utils.testing.imports as futi

IRIS_DATASET = fatf_datasets.load_iris()


@pytest.mark.skipif(SKLEARN_MISSING, reason='scikit-learn is not installed.')
def test_sklearn_imports(caplog):
    """
    Tests loading the package and using the explainers with sklearn missing.
    """
    missing_sklearn = ('The TabularBlimeyLime and TabularBlimeyTree surrogate '
                       'explainers require scikit-learn to be installed. '
                       'Since scikit-learn is missing, this functionality '
                       'will be disabled.')
    import_error_lime = ('The scikit-learn package is required to use '
                         'the TabularBlimeyLime explainer.')
    import_error_tree = ('The scikit-learn package is required to use '
                         'the TabularBlimeyTree explainer.')

    assert len(caplog.records) == 0
    assert 'fatf.transparency.predictions.surrogate_explainers' in sys.modules

    with futi.module_import_tester('sklearn', when_missing=True):
        with pytest.warns(UserWarning) as warning:
            importlib.reload(ftps)
        assert len(warning) == 1
        assert str(warning[0].message) == missing_sklearn
        #
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'ERROR'
        assert caplog.records[0].getMessage() == "No module named 'sklearn'"

        with pytest.raises(ImportError) as exin:
            ftps.TabularBlimeyLime(None, None)
        assert str(exin.value) == import_error_lime

        with pytest.raises(ImportError) as exin:
            ftps.TabularBlimeyTree(None, None)
        assert str(exin.value) == import_error_tree

    importlib.reload(ftps)
    assert len(caplog.records) == 1
    assert 'fatf.transparency.predictions.surrogate_explainers' in sys.modules


def test_input_is_valid():
    """
    Tests the ``_is_input_valid`` method.

    Tests the
    :func:`fatf.transparency.predictions.surrogate_explainers._is_input_valid`
    method.
    """
    dataset_incorrect_shape = ('The input dataset must be a 2-dimensional '
                               'numpy array.')
    dataset_type_error = ('The input dataset must only contain base types '
                          '(textual and/or numerical).')
    as_probabilistic_type_error = ('The as_probabilistic parameter has to '
                                   'be a boolean.')
    as_regressor_type_error = 'The as_regressor parameter has to be a boolean.'
    model_incompatible_reg = ('With as_regressor set to True the predictive '
                              'model needs to be capable of outputting '
                              'numerical predictions via a *predict* method, '
                              'which takes exactly one required parameter -- '
                              'data to be predicted -- and outputs a '
                              '1-dimensional array with numerical '
                              'predictions.')
    model_incompatible_model_np = ('With as_probabilistic set to False the '
                                   'predictive model needs to be capable of '
                                   'outputting (class) predictions via a '
                                   '*predict* method, which takes exactly one '
                                   'required parameter -- data to be '
                                   'predicted -- and outputs a 1-dimensional '
                                   'array with (class) predictions.')
    model_incompatible_model_p = ('With as_probabilistic set to True the '
                                  'predictive model needs to be capable of '
                                  'outputting probabilities via a '
                                  '*predict_proba* method, which takes '
                                  'exactly one required parameter -- data to '
                                  'be predicted -- and outputs a '
                                  '2-dimensional array with probabilities.')
    categorical_indices_index_error = ('The following indices are invalid for '
                                       'the input dataset: {}.')
    categorical_indices_type_error = ('The categorical_indices parameter must '
                                      'be a Python list or None.')
    categorical_indices_value_error = ('The categorical_indices list contains '
                                       'duplicated entries.')
    class_names_type_error_out = ('The class_names parameter must be a Python '
                                  'list or None.')
    class_names_value_error_empty = 'The class_names list cannot be empty.'
    class_names_value_error_dup = ('The class_names list contains '
                                   'duplicated entries.')
    class_names_type_error_in = ('All elements of the class_names list must '
                                 'be strings; *{}* is not.')
    class_number_type_error = ('The classes number parameter must be an '
                               'integer or None.')
    class_number_value_error = ('The number of classes cannot be smaller '
                                'than 2.')
    feature_names_type_error_out = ('The feature_names parameter must be a '
                                    'Python list or None.')
    feature_names_value_error_count = ('The length of feature_names must be '
                                       'equal to the number of features in '
                                       'the dataset ({}).')
    feature_names_value_error_dup = ('The feature_names list contains '
                                     'duplicated entries.')
    feature_names_type_error_in = ('All elements of the feature_names list '
                                   'have to be strings; *{}* is not.')
    unique_predictions_type_error_out = ('The unique_predictions parameter '
                                         'must be a Python list or None.')
    unique_predictions_value_error_empty = ('The unique_predictions list '
                                            'cannot be empty.')
    unique_predictions_value_error_dup = ('The unique_predictions list '
                                          'contains duplicated entries.')
    unique_predictions_type_error_in = ('One of the elements in the '
                                        'unique_predictions list is neither a '
                                        'string nor an integer.')

    with pytest.raises(IncorrectShapeError) as exin:
        ftps._input_is_valid(futt.LABELS, None, None, None, None, None, None,
                             None, None)
    assert str(exin.value) == dataset_incorrect_shape
    with pytest.raises(TypeError) as exin:
        array = np.array([[0, None], [0, 8]])
        ftps._input_is_valid(array, None, None, None, None, None, None, None,
                             None)
    assert str(exin.value) == dataset_type_error

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, None, None, None, None,
                             None, None, None, None)
    assert str(exin.value) == as_probabilistic_type_error
    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, None, True, None, None,
                             None, None, None, None)
    assert str(exin.value) == as_regressor_type_error

    model = futt.InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, True, True, None,
                             None, None, None, None)
    assert str(exin.value) == model_incompatible_reg
    with pytest.raises(IncompatibleModelError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, True, None,
                             None, None, None, None)
    assert str(exin.value) == model_incompatible_reg
    with pytest.raises(IncompatibleModelError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None, None)
    assert str(exin.value) == model_incompatible_model_np
    with pytest.raises(IncompatibleModelError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, True, False, None,
                             None, None, None, None)
    assert str(exin.value) == model_incompatible_model_p

    model = futt.NonProbabilisticModel(None)
    with pytest.raises(IncompatibleModelError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, True, False, None,
                             None, None, None, None)
    assert str(exin.value) == model_incompatible_model_p

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False, 'a',
                             None, None, None, None)
    assert str(exin.value) == categorical_indices_type_error
    with pytest.raises(IndexError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             ['a'], None, None, None, None)
    assert str(exin.value) == categorical_indices_index_error.format(
        np.array(['a']))
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             ['a', 'b', 'a', 'c'], None, None, None, None)
    assert str(exin.value) == categorical_indices_value_error

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, 'a', None, None, None)
    assert str(exin.value) == class_names_type_error_out
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, [], None, None, None)
    assert str(exin.value) == class_names_value_error_empty
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, ['a', 'b', 'a', 'c'], None, None, None)
    assert str(exin.value) == class_names_value_error_dup
    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, ['a', 0, 'b'], None, None, None)
    assert str(exin.value) == class_names_type_error_in.format(0)

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, 'a', None, None)
    assert str(exin.value) == class_number_type_error
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, 1, None, None)
    assert str(exin.value) == class_number_value_error

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, 'a', None)
    assert str(exin.value) == feature_names_type_error_out
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, ['a'], None)
    assert str(exin.value) == feature_names_value_error_count.format(4)
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, ['a', 'b', 'a', 'c'], None)
    assert str(exin.value) == feature_names_value_error_dup
    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, ['0', '1', 2, '3'], None)
    assert str(exin.value) == feature_names_type_error_in.format(2)

    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None, 'list')
    assert str(exin.value) == unique_predictions_type_error_out
    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None, [None, 's', 't'])
    assert str(exin.value) == unique_predictions_type_error_in
    with pytest.raises(TypeError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None,
                             ['s', 't', 'r', 1, 'n', 'g'])
    assert str(exin.value) == unique_predictions_type_error_in
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None, [])
    assert str(exin.value) == unique_predictions_value_error_empty
    with pytest.raises(ValueError) as exin:
        ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                             None, None, None, None, ['a', 'b', 'b', 'a'])
    assert str(exin.value) == unique_predictions_value_error_dup

    assert ftps._input_is_valid(futt.NUMERICAL_NP_ARRAY, model, False, False,
                                None, None, None, None, None)


class TestSurrogateTabularExplainer(object):
    """
    Tests the ``SurrogateTabularExplainer`` abstract class.

    Tests the :class:`fatf.transparency.predictions.surrogate_explainers.\
SurrogateTabularExplainer` abstract class.
    """

    class BrokenSurrogateTabularExplainer(ftps.SurrogateTabularExplainer):
        """
        A broken surrogate tabular explainer implementation.

        This class does not overwrite the ``explain_instance`` method.
        """

    class BaseSurrogateTabularExplainer(ftps.SurrogateTabularExplainer):
        """
        A dummy surrogate tabular explainer implementation.
        """

        def __init__(self,
                     dataset,
                     predictive_model,
                     as_probabilistic=True,
                     as_regressor=False,
                     categorical_indices=None,
                     class_names=None,
                     classes_number=None,
                     feature_names=None,
                     unique_predictions=None):
            """
            Dummy initialisation method.
            """
            super().__init__(dataset, predictive_model, as_probabilistic,
                             as_regressor, categorical_indices, class_names,
                             classes_number, feature_names, unique_predictions)

        def explain_instance(self, data_row):
            """
            Dummy ``explain_instance`` method.
            """
            self._explain_instance_input_is_valid(data_row)

    numerical_np_array_classifier = fum.KNN(k=3)
    numerical_np_array_classifier.fit(futt.NUMERICAL_NP_ARRAY, futt.LABELS)
    numerical_dummy_surrogate = BaseSurrogateTabularExplainer(
        futt.NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier,
        as_probabilistic=True,
        categorical_indices=[0],
        class_names=None,
        feature_names=['1', '2', '3', '4'])
    numerical_dummy_surrogate_reg = BaseSurrogateTabularExplainer(
        futt.NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier,
        as_probabilistic=True,
        as_regressor=True,
        categorical_indices=[0],
        class_names=['a', 'b', 'c'],
        feature_names=['1', '2', '3', '4'])

    numerical_struct_array_classifier = fum.KNN(k=3)
    numerical_struct_array_classifier.fit(futt.NUMERICAL_STRUCT_ARRAY,
                                          futt.LABELS)
    numerical_struct_dummy_surrogate = BaseSurrogateTabularExplainer(
        futt.NUMERICAL_STRUCT_ARRAY,
        numerical_struct_array_classifier,
        as_probabilistic=False,
        categorical_indices=['a', 'b'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=None,
        classes_number=3,
        unique_predictions=[0, 1, 2])

    categorical_np_array_classifier = fum.KNN(k=3)
    categorical_np_array_classifier.fit(futt.CATEGORICAL_NP_ARRAY, futt.LABELS)
    categorical_np_dummy_surrogate = BaseSurrogateTabularExplainer(
        futt.CATEGORICAL_NP_ARRAY,
        categorical_np_array_classifier,
        categorical_indices=[0, 1, 2])

    categorical_struct_array_classifier = fum.KNN(k=3)
    categorical_struct_array_classifier.fit(futt.CATEGORICAL_STRUCT_ARRAY,
                                            futt.LABELS)
    categorical_struct_dummy_surrogate = BaseSurrogateTabularExplainer(
        futt.CATEGORICAL_STRUCT_ARRAY,
        categorical_struct_array_classifier,
        categorical_indices=['a', 'b', 'c'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=['1', '2', '3'])

    mixed_classifier = fum.KNN(k=3)
    mixed_classifier.fit(futt.MIXED_ARRAY, futt.LABELS)
    mixed_dummy_surrogate = BaseSurrogateTabularExplainer(
        futt.MIXED_ARRAY,
        mixed_classifier,
        categorical_indices=['b', 'd'],
        feature_names=['num1', 'str1', 'num2', 'str2'])
    mixed_dummy_surrogate_reg = BaseSurrogateTabularExplainer(
        futt.MIXED_ARRAY,
        mixed_classifier,
        as_probabilistic=False,
        as_regressor=True,
        categorical_indices=['b', 'd'],
        class_names=['a', 'b', 'c'],
        classes_number=8,
        unique_predictions=['d', 'e'],
        feature_names=['num1', 'str1', 'num2', 'str2'])

    def test_surrogate_explainer_init(self, caplog):
        """
        Tests initialisation of ``SurrogateTabularExplainer`` class children.

        Tests the :func:`fatf.transparency.predictions.surrogate_explainers.\
SurrogateTabularExplainer.__init__` initialisation method.
        """
        abstract_method_error = ("Can't instantiate abstract class "
                                 '{} with abstract methods explain_instance')
        user_warning_features = (
            'Some of the string-based columns in the input dataset were not '
            'selected as categorical features via the categorical_indices '
            'parameter. String-based columns cannot be treated as numerical '
            'features, therefore they will be also treated as categorical '
            'features (in addition to the ones selected with the '
            'categorical_indices parameter).')
        user_warning_predictions_missing = (
            'The unique predictions ({}) were inferred from '
            'predicting the input dataset. Since this may not '
            'be accurate please consider specifying the list '
            'of unique predictions via the unique_predictions '
            'parameter.')
        user_warning_predictions_surplus = (
            'The unique_predictions provided by the user '
            'will be disregarded as the predictive_model '
            'is probabilistic (as_probabilistic=True).')
        user_warning_classes = (
            'The number of classes ({}) was inferred from '
            'predicting the input dataset. Since this may not '
            'be accurate please consider specifying the number '
            'of unique classes via the classes_number '
            'parameter.')

        debug_log_class_names = ('The classes number was taken from the '
                                 'length of the class_names list.')
        debug_log_unique_predictions = ('The classes number was taken from '
                                        'the length of the unique_predictions '
                                        'list.')

        classes_number_runtime_error = ('The user specified number of classes '
                                        '({}) for the provided probabilistic '
                                        'model is different than the number '
                                        'of columns ({}) in the probabilistic '
                                        'matrix output by the model.')
        unique_predictions_runtime_error_different = (
            'The predictive_model has output different classes ({} extra) '
            'than were specified by the unique_predictions parameter.')
        unique_predictions_runtime_error_number_i = (
            'The inferred number of unique predictions ({}) does not agree '
            'with the internal number of classes. Try providing the '
            'unique_predictions parameter to fix this issue.')
        unique_predictions_runtime_error_number_u = (
            'The user-specified number of unique predictions ({}) does not '
            'agree with the internal number of classes. (The length of the '
            'unique_predictions list is different than the classes_number '
            'parameter.)')
        class_names_value_error = ('The length of the class_names list does '
                                   'not agree with the number of classes '
                                   '({}).')

        unique_predictions = [0, 1, 2]

        assert len(caplog.records) == 0

        # Missing explain_instance method.
        with pytest.raises(TypeError) as exin:
            ftps.SurrogateTabularExplainer(futt.NUMERICAL_NP_ARRAY,
                                           self.numerical_np_array_classifier)
        assert str(exin.value) == abstract_method_error.format(
            'SurrogateTabularExplainer')
        #
        with pytest.raises(TypeError) as exin:
            self.BrokenSurrogateTabularExplainer(
                futt.NUMERICAL_NP_ARRAY, self.numerical_np_array_classifier)
        assert str(exin.value) == abstract_method_error.format(
            'BrokenSurrogateTabularExplainer')

        # ## Probabilistic models ########################################### #

        # A probabilistic model with the wrong number of user-specified classes
        with pytest.raises(RuntimeError) as exin:
            self.BaseSurrogateTabularExplainer(
                futt.NUMERICAL_NP_ARRAY,
                self.numerical_np_array_classifier,
                as_probabilistic=True,
                classes_number=4)
        assert str(exin.value) == classes_number_runtime_error.format(4, 3)

        # A probabilistic model with user-specified unique predictions
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.NUMERICAL_STRUCT_ARRAY,
                self.numerical_struct_array_classifier,
                as_probabilistic=True,
                unique_predictions=['a', 'b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning_predictions_surplus
        assert (surrogate_explainer.predictive_function  # yapf: disable
                == self.numerical_struct_array_classifier.predict_proba)
        assert surrogate_explainer.unique_predictions is None

        # ## Non-probabilistic models ####################################### #

        # Warning for a non-probabilistic model and missing classes number.
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False)
        assert (surrogate_explainer.predictive_function  # yapf: disable
                == self.categorical_np_array_classifier.predict)
        assert len(warning) == 2
        #
        assert str(warning[0].message) == user_warning_classes.format(3)
        assert surrogate_explainer.classes_number == 3
        # Warning for a non-probabilistic model and missing unique predictions.
        assert (str(warning[1].message)  # yapf: disable
                == user_warning_predictions_missing.format(unique_predictions))
        assert np.array_equal(surrogate_explainer.unique_predictions,
                              unique_predictions)
        assert (surrogate_explainer.class_names  # yapf: disable
                == ['class 0', 'class 1', 'class 2'])

        # Error for a non-probabilistic model and an incorrect number of
        # unique predictions inferred from the model and the data.
        with pytest.raises(RuntimeError) as exin:
            self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False,
                classes_number=4)
        assert (str(exin.value)  # yapf: disable
                == unique_predictions_runtime_error_number_i.format(3))

        # Error for a non-probabilistic model and an incorrect number of
        # unique predictions given by the user.
        with pytest.raises(RuntimeError) as exin:
            self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False,
                classes_number=3,
                unique_predictions=[1, 2])
        assert (str(exin.value)  # yapf: disable
                == unique_predictions_runtime_error_number_u.format(2))

        # Incorrect unique predictions detected for a non-probabilistic model.
        with pytest.raises(RuntimeError) as exin:
            self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False,
                classes_number=3,
                unique_predictions=[1, 2, 3])
        assert (str(exin.value)  # yapf: disable
                == unique_predictions_runtime_error_different.format([0]))

        # Debug log for a non-probabilistic model and missing classes number
        # inferred from the unique_predictions list.
        assert len(caplog.records) == 0
        surrogate_explainer = self.BaseSurrogateTabularExplainer(
            futt.CATEGORICAL_NP_ARRAY,
            self.categorical_np_array_classifier,
            as_probabilistic=False,
            unique_predictions=[0, 1, 2])
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'DEBUG'
        assert caplog.records[0].getMessage() == debug_log_unique_predictions
        assert surrogate_explainer.classes_number == 3
        assert (surrogate_explainer.class_names  # yapf: disable
                == ['class 0', 'class 1', 'class 2'])

        # Debug log for a non-probabilistic model and missing classes number
        # inferred from the class names.
        assert len(caplog.records) == 1
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False,
                class_names=['my class 0', 'my class 1', 'my class 2'])
        assert len(warning) == 1
        assert (str(warning[0].message)  # yapf: disable
                == user_warning_predictions_missing.format(unique_predictions))
        assert np.array_equal(surrogate_explainer.unique_predictions,
                              unique_predictions)
        #
        assert len(caplog.records) == 2
        assert caplog.records[1].levelname == 'DEBUG'
        assert caplog.records[1].getMessage() == debug_log_class_names
        assert surrogate_explainer.classes_number == 3
        #
        assert np.array_equal(surrogate_explainer.feature_names,
                              ['feature 0', 'feature 1', 'feature 2'])

        #######################################################################

        # Class names error
        with pytest.raises(ValueError) as exin:
            self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=False,
                unique_predictions=[0, 1, 2],
                classes_number=3,
                class_names=['mc 1', 'mc 2', 'mc 3', 'mc 4'])
        assert str(exin.value) == class_names_value_error.format(3)

        # Warning for handling categorical indices.
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_NP_ARRAY,
                self.categorical_np_array_classifier,
                as_probabilistic=True,
                categorical_indices=[0])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning_features
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              [0, 1, 2])
        assert np.array_equal(surrogate_explainer.numerical_indices, [])
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.CATEGORICAL_STRUCT_ARRAY,
                self.categorical_struct_array_classifier,
                as_probabilistic=True,
                categorical_indices=['a'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning_features
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              ['a', 'b', 'c'])
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateTabularExplainer(
                futt.MIXED_ARRAY,
                self.mixed_classifier,
                as_probabilistic=True,
                categorical_indices=['b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning_features
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              ['b', 'd'])

        #######################################################################

        # Validate class attributes
        assert np.array_equal(self.numerical_dummy_surrogate.dataset,
                              futt.NUMERICAL_NP_ARRAY)
        assert not self.numerical_dummy_surrogate.is_structured
        assert self.numerical_dummy_surrogate.column_indices == [0, 1, 2, 3]
        assert self.numerical_dummy_surrogate.categorical_indices == [0]
        assert self.numerical_dummy_surrogate.numerical_indices == [1, 2, 3]
        assert self.numerical_dummy_surrogate.as_probabilistic
        assert not self.numerical_dummy_surrogate.as_regressor
        # yapf: disable
        assert (self.numerical_dummy_surrogate.predictive_model
                == self.numerical_np_array_classifier)
        assert (self.numerical_dummy_surrogate.predictive_function
                == self.numerical_np_array_classifier.predict_proba)
        assert self.numerical_dummy_surrogate.classes_number == 3
        assert (self.numerical_dummy_surrogate.class_names
                == ['class 0', 'class 1', 'class 2'])
        assert (self.numerical_dummy_surrogate.feature_names
                == ['1', '2', '3', '4'])
        assert self.numerical_dummy_surrogate.unique_predictions is None

        assert np.array_equal(self.numerical_dummy_surrogate_reg.dataset,
                              futt.NUMERICAL_NP_ARRAY)
        assert not self.numerical_dummy_surrogate_reg.is_structured
        assert (self.numerical_dummy_surrogate_reg.column_indices
                == [0, 1, 2, 3])
        assert self.numerical_dummy_surrogate_reg.categorical_indices == [0]
        assert (self.numerical_dummy_surrogate_reg.numerical_indices
                == [1, 2, 3])
        assert self.numerical_dummy_surrogate_reg.as_probabilistic
        assert self.numerical_dummy_surrogate_reg.as_regressor
        assert (self.numerical_dummy_surrogate_reg.predictive_model
                == self.numerical_np_array_classifier)
        assert (self.numerical_dummy_surrogate_reg.predictive_function
                == self.numerical_np_array_classifier.predict)
        assert self.numerical_dummy_surrogate_reg.classes_number is None
        assert self.numerical_dummy_surrogate_reg.class_names is None
        assert (self.numerical_dummy_surrogate_reg.feature_names
                == ['1', '2', '3', '4'])
        assert self.numerical_dummy_surrogate_reg.unique_predictions is None

        assert np.array_equal(self.numerical_struct_dummy_surrogate.dataset,
                              futt.NUMERICAL_STRUCT_ARRAY)
        assert self.numerical_struct_dummy_surrogate.is_structured
        assert (self.numerical_struct_dummy_surrogate.column_indices
                == ['a', 'b', 'c', 'd'])
        assert (self.numerical_struct_dummy_surrogate.categorical_indices
                == ['a', 'b'])
        assert (self.numerical_struct_dummy_surrogate.numerical_indices
                == ['c', 'd'])
        assert not self.numerical_struct_dummy_surrogate.as_probabilistic
        assert not self.numerical_struct_dummy_surrogate.as_regressor
        assert (self.numerical_struct_dummy_surrogate.predictive_model
                == self.numerical_struct_array_classifier)
        assert (self.numerical_struct_dummy_surrogate.predictive_function
                == self.numerical_struct_array_classifier.predict)
        assert self.numerical_struct_dummy_surrogate.classes_number == 3
        assert (self.numerical_struct_dummy_surrogate.class_names
                == ['class1', 'class2', 'class3'])
        assert (self.numerical_struct_dummy_surrogate.feature_names
                == ['feature 0', 'feature 1', 'feature 2', 'feature 3'])
        assert (self.numerical_struct_dummy_surrogate.unique_predictions
                == [0, 1, 2])

        assert np.array_equal(self.categorical_np_dummy_surrogate.dataset,
                              futt.CATEGORICAL_NP_ARRAY)
        assert not self.categorical_np_dummy_surrogate.is_structured
        assert self.categorical_np_dummy_surrogate.column_indices == [0, 1, 2]
        assert (self.categorical_np_dummy_surrogate.categorical_indices
                == [0, 1, 2])
        assert self.categorical_np_dummy_surrogate.numerical_indices == []
        assert self.categorical_np_dummy_surrogate.as_probabilistic
        assert not self.categorical_np_dummy_surrogate.as_regressor
        assert (self.categorical_np_dummy_surrogate.predictive_model
                == self.categorical_np_array_classifier)
        assert (self.categorical_np_dummy_surrogate.predictive_function
                == self.categorical_np_array_classifier.predict_proba)
        assert self.categorical_np_dummy_surrogate.classes_number == 3
        assert (self.categorical_np_dummy_surrogate.class_names
                == ['class 0', 'class 1', 'class 2'])
        assert (self.categorical_np_dummy_surrogate.feature_names
                == ['feature 0', 'feature 1', 'feature 2'])
        assert self.categorical_np_dummy_surrogate.unique_predictions is None

        assert np.array_equal(self.categorical_struct_dummy_surrogate.dataset,
                              futt.CATEGORICAL_STRUCT_ARRAY)
        assert self.categorical_struct_dummy_surrogate.is_structured
        assert (self.categorical_struct_dummy_surrogate.column_indices
                == ['a', 'b', 'c'])
        assert (self.categorical_struct_dummy_surrogate.categorical_indices
                == ['a', 'b', 'c'])
        assert self.categorical_struct_dummy_surrogate.numerical_indices == []
        assert self.categorical_struct_dummy_surrogate.as_probabilistic
        assert not self.categorical_struct_dummy_surrogate.as_regressor
        assert (self.categorical_struct_dummy_surrogate.predictive_model
                == self.categorical_struct_array_classifier)
        assert (self.categorical_struct_dummy_surrogate.predictive_function
                == self.categorical_struct_array_classifier.predict_proba)
        assert self.categorical_struct_dummy_surrogate.classes_number == 3
        assert (self.categorical_struct_dummy_surrogate.class_names
                == ['class1', 'class2', 'class3'])
        assert (self.categorical_struct_dummy_surrogate.feature_names
                == ['1', '2', '3'])
        assert (self.categorical_struct_dummy_surrogate.unique_predictions
                is None)

        assert np.array_equal(self.mixed_dummy_surrogate.dataset,
                              futt.MIXED_ARRAY)
        assert self.mixed_dummy_surrogate.is_structured
        assert (self.mixed_dummy_surrogate.column_indices
                == ['a', 'b', 'c', 'd'])
        assert self.mixed_dummy_surrogate.categorical_indices == ['b', 'd']
        assert self.mixed_dummy_surrogate.numerical_indices == ['a', 'c']
        assert self.mixed_dummy_surrogate.as_probabilistic
        assert not self.mixed_dummy_surrogate.as_regressor
        assert (self.mixed_dummy_surrogate.predictive_model
                == self.mixed_classifier)
        assert (self.mixed_dummy_surrogate.predictive_function
                == self.mixed_classifier.predict_proba)
        assert self.mixed_dummy_surrogate.classes_number == 3
        assert (self.mixed_dummy_surrogate.class_names
                == ['class 0', 'class 1', 'class 2'])
        assert (self.mixed_dummy_surrogate.feature_names
                == ['num1', 'str1', 'num2', 'str2'])
        assert self.mixed_dummy_surrogate.unique_predictions is None

        assert np.array_equal(self.mixed_dummy_surrogate_reg.dataset,
                              futt.MIXED_ARRAY)
        assert self.mixed_dummy_surrogate_reg.is_structured
        assert (self.mixed_dummy_surrogate_reg.column_indices
                == ['a', 'b', 'c', 'd'])
        assert self.mixed_dummy_surrogate_reg.categorical_indices == ['b', 'd']
        assert self.mixed_dummy_surrogate_reg.numerical_indices == ['a', 'c']
        assert not self.mixed_dummy_surrogate_reg.as_probabilistic
        assert self.mixed_dummy_surrogate_reg.as_regressor
        assert (self.mixed_dummy_surrogate_reg.predictive_model
                == self.mixed_classifier)
        assert (self.mixed_dummy_surrogate_reg.predictive_function
                == self.mixed_classifier.predict)
        assert self.mixed_dummy_surrogate_reg.classes_number is None
        assert self.mixed_dummy_surrogate_reg.class_names is None
        assert (self.mixed_dummy_surrogate_reg.feature_names
                == ['num1', 'str1', 'num2', 'str2'])
        assert self.mixed_dummy_surrogate_reg.unique_predictions is None
        # yapf: enable

        assert len(caplog.records) == 2

    def test_explain_instance_validation(self):
        """
        Tests the ``_explain_instance_is_valid`` method.

        Tests the :func:`fatf.transparency.predictions.surrogate_explainers.\
SurrogateTabularExplainer._explain_instance_is_valid` method.
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
            self.numerical_dummy_surrogate.explain_instance(
                futt.NUMERICAL_NP_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                futt.NUMERICAL_STRUCT_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row

        # data_row type
        with pytest.raises(TypeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(
                np.array(['a', 'b', 'c', 'd']))
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                futt.MIXED_ARRAY[0])
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.categorical_np_dummy_surrogate.explain_instance(
                np.array([0.1]))
        assert str(exin.value) == type_error_data_row
        # Structured, too short
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                futt.MIXED_ARRAY[['a', 'b']][0])
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


@pytest.mark.skipif(SKLEARN_MISSING, reason='scikit-learn is not installed.')
class TestTabularBlimeyLime(object):
    """
    Tests the ``TabularBlimeyLime`` class.

    Tests the :class:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyLime` class.
    """
    if not SKLEARN_MISSING:
        numerical_np_array_classifier = fum.KNN(k=3)
        numerical_np_array_classifier.fit(futt.NUMERICAL_NP_ARRAY, futt.LABELS)
        numerical_np_tabular_lime = ftps.TabularBlimeyLime(
            futt.NUMERICAL_NP_ARRAY, numerical_np_array_classifier)
        numerical_np_tabular_lime_reg = ftps.TabularBlimeyLime(
            futt.NUMERICAL_NP_ARRAY,
            numerical_np_array_classifier,
            as_regressor=True)

        wide_data = np.concatenate(2 * [futt.NUMERICAL_NP_ARRAY], axis=1)
        numerical_np_array_classifier_wide = fum.KNN(k=3)
        numerical_np_array_classifier_wide.fit(wide_data, futt.LABELS)
        numerical_np_tabular_lime_wide_reg = ftps.TabularBlimeyLime(
            wide_data, numerical_np_array_classifier_wide, as_regressor=True)

        numerical_struct_array_classifier = fum.KNN(k=3)
        numerical_struct_array_classifier.fit(futt.NUMERICAL_STRUCT_ARRAY,
                                              futt.LABELS)
        numerical_struct_cat_tabular_lime = ftps.TabularBlimeyLime(
            futt.NUMERICAL_STRUCT_ARRAY,
            numerical_struct_array_classifier,
            categorical_indices=['a', 'b'])

        categorical_array_classifier = fum.KNN(k=3)
        categorical_array_classifier.fit(futt.CATEGORICAL_NP_ARRAY,
                                         futt.LABELS)
        categorical_np_lime = ftps.TabularBlimeyLime(
            futt.CATEGORICAL_NP_ARRAY,
            categorical_array_classifier,
            categorical_indices=[0, 1, 2])

        iris_classifier = fum.KNN(k=3)
        iris_classifier.fit(IRIS_DATASET['data'], IRIS_DATASET['target'])
        iris_lime = ftps.TabularBlimeyLime(
            IRIS_DATASET['data'],
            iris_classifier,
            class_names=IRIS_DATASET['target_names'].tolist(),
            feature_names=IRIS_DATASET['feature_names'].tolist())

    def test_init(self):
        """
        Tests the ``TabularBlimeyLime`` initialisation.

        Tests the initialisation of the :class:`fatf.transparency.predictions.\
surrogate_explainers.TabularBlimeyLime` class.
        """
        # Test class inheritance
        # yapf: disable
        assert (
            self.numerical_np_tabular_lime.__class__.__bases__[0].__name__
            == 'SurrogateTabularExplainer'
        )
        # yapf: enable

        numerical_bin_sampling_values = {
            0: {  # Index 0
                0: (0.0, 0.0, 0.0, 0.0),
                1: (0.0, 0.5, np.nan, np.nan),
                2: (0.5, 1.0, 1.0, 0.0),
                3: (1.0, 2.0, 2.0, 0.0)
            },
            1: {  # Index 1
                0: (0.0, 0.0, 0.0, 0.0),
                1: (0.0, 0.5, np.nan, np.nan),
                2: (0.5, 1.0, 1.0, 0.0),
                3: (1.0, np.inf, np.nan, np.nan)
            },
            2: {  # Index 2
                0: (0.03, 0.073, 0.05, 0.02),
                1: (0.073, 0.22, 0.08, 0.0),
                2: (0.22, 0.638, 0.36, 0.0),
                3: (0.638, 0.99, 0.86, 0.13)
            },  # Index 3
            3: {
                0: (0.21, 0.337, 0.25, 0.04),
                1: (0.337, 0.585, 0.48, 0.0),
                2: (0.585, 0.788, 0.69, 0.0),
                3: (0.7875, 0.89, 0.855, 0.035)
            }
        }
        numerical_struct_sampling_values = {
            'c': numerical_bin_sampling_values[2],
            'd': numerical_bin_sampling_values[3]
        }
        iris_lime_sampling_values = {
            0: {
                0: (4.3, 5.1, 4.856, 0.229),
                1: (5.1, 5.8, 5.559, 0.185),
                2: (5.8, 6.4, 6.189, 0.163),
                3: (6.4, 7.9, 6.971, 0.412)
            },
            1: {
                0: (2.0, 2.8, 2.585, 0.208),
                1: (2.8, 3.0, 2.972, 0.045),
                2: (3.0, 3.3, 3.183, 0.073),
                3: (3.3, 4.4, 3.638, 0.252)
            },
            2: {
                0: (1.0, 1.6, 1.420, 0.134),
                1: (1.6, 4.35, 3.474, 0.890),
                2: (4.35, 5.1, 4.766, 0.243),
                3: (5.1, 6.9, 5.826, 0.437)
            },
            3: {
                0: (0.1, 0.3, 0.205, 0.054),
                1: (0.3, 1.3, 1.003, 0.342),
                2: (1.3, 1.8, 1.595, 0.157),
                3: (1.8, 2.5, 2.171, 0.187)
            }
        }

        # yapf: disable
        assert (self.numerical_np_tabular_lime.numerical_indices
                == [0, 1, 2, 3])
        # yapf: enable
        assert self.numerical_np_tabular_lime.categorical_indices == []
        assert isinstance(self.numerical_np_tabular_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert isinstance(self.numerical_np_tabular_lime.augmenter,
                          fuda.NormalSampling)
        assert futt.is_explanation_equal_dict(
            self.numerical_np_tabular_lime.bin_sampling_values,
            numerical_bin_sampling_values)

        # yapf: disable
        assert (self.numerical_struct_cat_tabular_lime.numerical_indices
                == ['c', 'd'])
        assert (self.numerical_struct_cat_tabular_lime.categorical_indices
                == ['a', 'b'])
        # yapf: enable
        assert isinstance(self.numerical_struct_cat_tabular_lime.augmenter,
                          fuda.NormalSampling)
        assert isinstance(self.numerical_struct_cat_tabular_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert futt.is_explanation_equal_dict(
            self.numerical_struct_cat_tabular_lime.bin_sampling_values,
            numerical_struct_sampling_values)

        assert self.categorical_np_lime.numerical_indices == []
        assert self.categorical_np_lime.categorical_indices == [0, 1, 2]
        assert isinstance(self.categorical_np_lime.augmenter,
                          fuda.NormalSampling)
        assert isinstance(self.categorical_np_lime.discretiser,
                          fudd.QuartileDiscretiser)
        assert self.categorical_np_lime.bin_sampling_values == {}

        assert self.iris_lime.numerical_indices == [0, 1, 2, 3]
        assert self.iris_lime.categorical_indices == []
        assert isinstance(self.iris_lime.augmenter, fuda.NormalSampling)
        assert isinstance(self.iris_lime.discretiser, fudd.QuartileDiscretiser)
        assert futt.is_explanation_equal_dict(
            self.iris_lime.bin_sampling_values, iris_lime_sampling_values)

    def test_explain_instance_input_is_valid(self):
        """
        Tests the ``_explain_instance_input_is_valid`` method.

        Tests :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyLime._explain_instance_input_is_valid` method.
        """
        explained_class_type = ('The explained_class parameter must be either '
                                'None, a string or an integer.')
        samples_number_type = ('The samples_number parameter must be an '
                               'integer.')
        samples_number_value = ('The samples_number parameter must be a '
                                'positive integer (larger than 0).')
        features_number_type = ('The features_number parameter must either be '
                                'None or an integer.')
        features_number_value = ('The features_number parameter must be a '
                                 'positive integer (larger than 0).')
        kernel_width_type = ('The kernel_width parameter must either be None '
                             'or a float.')
        kernel_width_value = ('The kernel_width parameter must be a positive '
                              'float (larger than 0).')
        return_models_type = 'The return_models parameter must be a boolean.'

        instance = futt.NUMERICAL_NP_ARRAY[0]

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, 7.0, None, None, None, None)
        assert str(exin.value) == explained_class_type

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 'a', None, None, None)
        assert str(exin.value) == samples_number_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, -1, None, None, None)
        assert str(exin.value) == samples_number_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 1, 'a', None, None)
        assert str(exin.value) == features_number_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 1, -1, None, None)
        assert str(exin.value) == features_number_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 1, None, 'a', None)
        assert str(exin.value) == kernel_width_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 1, None, -1, None)
        assert str(exin.value) == kernel_width_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_lime._explain_instance_input_is_valid(
                instance, None, 1, None, None, None)
        assert str(exin.value) == return_models_type

        # All good
        assert self.numerical_np_tabular_lime._explain_instance_input_is_valid(
            instance, None, 10, None, None, False)
        assert self.numerical_np_tabular_lime._explain_instance_input_is_valid(
            instance, None, 10, None, 1, False)
        assert self.numerical_np_tabular_lime._explain_instance_input_is_valid(
            instance, None, 10, 3, 0.1, False)

    def test_undiscretise_data(self):
        """
        Tests the ``_undiscretise_data`` method.

        Tests :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyLime._undiscretise_data` method.
        """
        fatf.setup_random_seed()

        # Impossible data points
        impossible_point = 'No empirical mean for a bin without data points.'
        with pytest.raises(AssertionError) as exin:
            self.numerical_np_tabular_lime._undiscretise_data(
                np.array([[1, 0, 0, 0]]))
        assert str(exin.value) == impossible_point
        with pytest.raises(AssertionError) as exin:
            self.numerical_np_tabular_lime._undiscretise_data(
                np.array([[1, 1, 0, 0]]))
        assert str(exin.value) == impossible_point
        with pytest.raises(AssertionError) as exin:
            self.numerical_np_tabular_lime._undiscretise_data(
                np.array([[0, 1, 0, 0]]))
        assert str(exin.value) == impossible_point
        with pytest.raises(AssertionError) as exin:
            self.numerical_np_tabular_lime._undiscretise_data(
                np.array([[0, 3, 0, 0]]))
        assert str(exin.value) == impossible_point

        udata = self.numerical_np_tabular_lime._undiscretise_data(
            np.array([[0, 0, 0, 0]]))
        assert np.allclose(udata, [[0, 0, 0.046, 0.314]], atol=1e-3)

        dtype = futt.NUMERICAL_STRUCT_ARRAY.dtype
        udata = self.numerical_struct_cat_tabular_lime._undiscretise_data(
            np.array([(0, 0, 0, 0)], dtype=dtype))
        assert np.allclose(
            fuat.as_unstructured(udata), [[0, 0, 0.059, 0.266]], atol=1e-3)

    def test_explain_instance_errors(self):
        """
        Tests errors and exceptions in the ``explain_instance`` method.

        Tests errors and exceptions in the :func:`fatf.transparency.\
predictions.surrogate_explainers.TabularBlimeyLime.explain_instance` method.
        """
        explain_class_value_error1 = ('The *{}* explained class name was not '
                                      'recognised. The following class names '
                                      'are allowed: {}.')
        explain_class_value_error2 = ('The explained class index is out of '
                                      'the allowed range: 0 to {} (there are '
                                      '{} classes altogether).')

        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0], explained_class='class 3')
        assert str(exin.value) == explain_class_value_error1.format(
            'class 3', ['class 0', 'class 1', 'class 2'])

        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_lime.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0], explained_class=3)
        assert str(exin.value) == explain_class_value_error2.format(2, 3)

    def test_explain_instance(self, caplog):
        """
        Tests the ``explain_instance`` method.

        Tests :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyLime.explain_instance` method.
        """
        assert len(caplog.records) == 0
        fatf.setup_random_seed()
        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage() == ('Seeding RNGs using the '
                                                  'system variable.')
        assert caplog.records[1].levelname == 'INFO'
        assert caplog.records[1].getMessage() == 'Seeding RNGs with 42.'

        log_info_forward_selection = ('Selecting {} features with forward '
                                      'selection.')
        log_info_highest_weights = ('Selecting {} features with highest '
                                    'weights.')

        numerical_np_explanation = {
            'class 0': {
                '*feature 0* <= 0.00': -0.338,
                '*feature 1* <= 0.00': 0.255,
                '0.07 < *feature 2* <= 0.22': 0.116,
                '0.58 < *feature 3* <= 0.79': 0.069
            },
            'class 1': {
                '*feature 0* <= 0.00': 0.029,
                '*feature 1* <= 0.00': -0.160,
                '0.07 < *feature 2* <= 0.22': -0.069,
                '0.58 < *feature 3* <= 0.79': 0.018
            },
            'class 2': {
                '*feature 0* <= 0.00': 0.309,
                '*feature 1* <= 0.00': -0.096,
                '0.07 < *feature 2* <= 0.22': -0.047,
                '0.58 < *feature 3* <= 0.79': -0.087
            }
        }
        numerical_np_explanation_reg = {
            '*feature 0* <= 0.00': 1.058,
            '*feature 1* <= 0.00': -0.659,
            '0.07 < *feature 2* <= 0.22': -0.065,
            '0.58 < *feature 3* <= 0.79': -0.015
        }
        numerical_np_explanation_wide_reg = {
            '*feature 0* <= 0.00': 0.571,
            '*feature 1* <= 0.00': -0.626,
            '0.07 < *feature 2* <= 0.22': -0.328,
            '*feature 4* <= 0.00': 0.442,
            '*feature 5* <= 0.00': -0.265,
            '0.07 < *feature 6* <= 0.22': 0.371,
            '0.58 < *feature 7* <= 0.79': 0.230
        }
        numerical_np_explanation_wide_reg_all = {
            '*feature 0* <= 0.00': 0.752,
            '*feature 1* <= 0.00': -0.260,
            '0.07 < *feature 2* <= 0.22': 0.246,
            '0.58 < *feature 3* <= 0.79': -0.464,
            '*feature 4* <= 0.00': 0.811,
            '*feature 5* <= 0.00': -0.305,
            '0.07 < *feature 6* <= 0.22': -0.036,
            '0.58 < *feature 7* <= 0.79': 0.096
        }
        numerical_struct_explanation = {
            'class 0': {
                '*feature 0* = 0': -0.299,
                '*feature 1* = 0': 0.246
            },
            'class 1': {
                '*feature 1* = 0': -0.139,
                '0.07 < *feature 2* <= 0.22': -0.148
            },
            'class 2': {
                '*feature 0* = 0': 0.234,
                '*feature 1* = 0': -0.061
            }
        }
        categorical_np_explanation = {
            'class 0': {
                '*feature 0* = a': 0.258,
                '*feature 1* = b': -0.201
            },
            'class 1': {
                '*feature 0* = a': -0.490,
                '*feature 1* = b': -0.105
            },
            'class 2': {
                '*feature 0* = a': 0.232,
                '*feature 1* = b': 0.306
            }
        }
        iris_explanation = {
            'setosa': {
                '*petal length (cm)* <= 1.60': 0.780,
                '3.30 < *sepal width (cm)*': 0.066
            },
            'versicolor': {
                '*petal length (cm)* <= 1.60': -0.543,
                '*petal width (cm)* <= 0.30': 0.195
            },
            'virginica': {
                '*petal length (cm)* <= 1.60': -0.237,
                '*petal width (cm)* <= 0.30': -0.240
            }
        }

        assert len(caplog.records) == 2
        # Probabilistic
        explanation = self.numerical_np_tabular_lime.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            features_number=4,
            kernel_width=None)
        assert futt.is_explanation_equal_dict(
            numerical_np_explanation, explanation, atol=1e-3)
        assert len(caplog.records) == 3
        assert caplog.records[2].levelname == 'INFO'
        assert (caplog.records[2].getMessage()  # yapf: disable
                == log_info_forward_selection.format(4))

        explanation = self.numerical_struct_cat_tabular_lime.explain_instance(
            futt.NUMERICAL_STRUCT_ARRAY[0],
            samples_number=50,
            features_number=2,
            kernel_width=None)
        assert futt.is_explanation_equal_dict(
            numerical_struct_explanation, explanation, atol=1e-3)
        assert len(caplog.records) == 4
        assert caplog.records[3].levelname == 'INFO'
        assert (caplog.records[3].getMessage()  # yapf: disable
                == log_info_forward_selection.format(2))

        explanation = self.categorical_np_lime.explain_instance(
            futt.CATEGORICAL_NP_ARRAY[0],
            samples_number=50,
            features_number=2,
            kernel_width=None)
        assert futt.is_explanation_equal_dict(
            categorical_np_explanation, explanation, atol=1e-3)
        assert len(caplog.records) == 5
        assert caplog.records[4].levelname == 'INFO'
        assert (caplog.records[4].getMessage()  # yapf: disable
                == log_info_forward_selection.format(2))

        explanation, models = self.iris_lime.explain_instance(
            IRIS_DATASET['data'][0],
            samples_number=50,
            features_number=2,
            kernel_width=None,
            return_models=True)
        assert futt.is_explanation_equal_dict(
            iris_explanation, explanation, atol=1e-3)
        for key, model in models.items():
            assert np.allclose(
                sorted(list(iris_explanation[key].values())),
                sorted(model.coef_.tolist()),
                atol=1e-3)
        assert len(caplog.records) == 6
        assert caplog.records[5].levelname == 'INFO'
        assert (caplog.records[5].getMessage()  # yapf: disable
                == log_info_forward_selection.format(2))

        explanation = self.iris_lime.explain_instance(
            IRIS_DATASET['data'][0],
            samples_number=50,
            features_number=1,
            kernel_width=1,
            explained_class='setosa')
        explanation_ = {'setosa': {'*petal length (cm)* <= 1.60': 0.666}}
        assert futt.is_explanation_equal_dict(
            explanation, explanation_, atol=1e-3)
        assert len(caplog.records) == 7
        assert caplog.records[6].levelname == 'INFO'
        assert (caplog.records[6].getMessage()  # yapf: disable
                == log_info_forward_selection.format(1))

        explanation = self.iris_lime.explain_instance(
            IRIS_DATASET['data'][0],
            samples_number=50,
            features_number=1,
            kernel_width=1,
            explained_class=1)
        explanation_ = {'versicolor': {'*petal length (cm)* <= 1.60': -0.357}}
        assert futt.is_explanation_equal_dict(
            explanation, explanation_, atol=1e-3)
        assert len(caplog.records) == 8
        assert caplog.records[7].levelname == 'INFO'
        assert (caplog.records[7].getMessage()  # yapf: disable
                == log_info_forward_selection.format(1))

        # Regression
        explanation = self.numerical_np_tabular_lime_reg.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            features_number=4,
            kernel_width=None)
        assert futt.is_explanation_equal_dict(
            {'': numerical_np_explanation_reg}, {'': explanation}, atol=1e-3)
        assert len(caplog.records) == 9
        assert caplog.records[8].levelname == 'INFO'
        assert (caplog.records[8].getMessage()  # yapf: disable
                == log_info_forward_selection.format(4))

        # Weight-based feature selection
        explanation = self.numerical_np_tabular_lime_wide_reg.explain_instance(
            self.wide_data[0], samples_number=50, features_number=7)
        assert futt.is_explanation_equal_dict(
            {'': numerical_np_explanation_wide_reg}, {'': explanation},
            atol=1e-3)
        assert len(caplog.records) == 10
        assert caplog.records[9].levelname == 'INFO'
        assert (caplog.records[9].getMessage()  # yapf: disable
                == log_info_highest_weights.format(7))
        #
        explanation = self.numerical_np_tabular_lime_wide_reg.explain_instance(
            self.wide_data[0], samples_number=50)
        assert futt.is_explanation_equal_dict(
            {'': numerical_np_explanation_wide_reg_all}, {'': explanation},
            atol=1e-3)
        assert len(caplog.records) == 11
        assert caplog.records[10].levelname == 'INFO'
        assert (caplog.records[10].getMessage()  # yapf: disable
                == log_info_highest_weights.format(8))


def map_target(target):
    """
    Maps 0->'a', 1->'b' and 2->'c'.
    """
    categorical_target = []
    for i in target:
        if i == 0:
            categorical_target.append('a')
        elif i == 1:
            categorical_target.append('b')
        elif i == 2:
            categorical_target.append('c')
        else:
            assert False, 'Only 0, 1 and 2 allowed.'  # pragma: nocover
    return np.array(categorical_target)


@pytest.mark.skipif(SKLEARN_MISSING, reason='scikit-learn is not installed.')
class TestTabularBlimeyTree(object):
    """
    Tests the ``TabularBlimeyTree`` class.

    Tests the :class:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyTree` class.
    """
    if not SKLEARN_MISSING:
        numerical_np_array_classifier = fum.KNN(k=3)
        numerical_np_array_classifier.fit(futt.NUMERICAL_NP_ARRAY, futt.LABELS)
        numerical_np_tabular_blimey = ftps.TabularBlimeyTree(
            futt.NUMERICAL_NP_ARRAY, numerical_np_array_classifier)
        numerical_np_tabular_blimey_reg = ftps.TabularBlimeyTree(
            futt.NUMERICAL_NP_ARRAY,
            numerical_np_array_classifier,
            as_regressor=True)

        numerical_np_array_classifier_noprob = fum.KNN(k=3)
        numerical_np_array_classifier_noprob.fit(futt.NUMERICAL_NP_ARRAY,
                                                 map_target(futt.LABELS))
        numerical_np_tabular_blimey_noprob = ftps.TabularBlimeyTree(
            futt.NUMERICAL_NP_ARRAY,
            numerical_np_array_classifier_noprob,
            as_probabilistic=False,
            classes_number=3,
            unique_predictions=['a', 'b', 'c'],
            class_names=['class 0', 'class 1', 'class 2'])

        numerical_struct_array_classifier = fum.KNN(k=3)
        numerical_struct_array_classifier.fit(futt.NUMERICAL_STRUCT_ARRAY,
                                              futt.LABELS)
        numerical_np_cat_tabular_blimey = ftps.TabularBlimeyTree(
            futt.NUMERICAL_NP_ARRAY,
            numerical_np_array_classifier,
            categorical_indices=[0, 1])

        iris_classifier = fum.KNN(k=3)
        iris_classifier.fit(IRIS_DATASET['data'], IRIS_DATASET['target'])
        iris_blimey = ftps.TabularBlimeyTree(
            IRIS_DATASET['data'],
            iris_classifier,
            class_names=IRIS_DATASET['target_names'].tolist(),
            feature_names=IRIS_DATASET['feature_names'].tolist())

    def test_init(self):
        """
        Tests the initialisation of the ``TabularBlimeyTree`` class.

        Tests the initialisation of the :class:`fatf.transparency.predictions.\
surrogate_explainers.TabularBlimeyTree` class.
        """
        # Test class inheritance
        # yapf: disable
        assert (
            self.numerical_np_tabular_blimey.__class__.__bases__[0].__name__
            == 'SurrogateTabularExplainer'
        )
        # yapf: enable

        string_array_error = ('The TabularBlimeyTree explainer does not '
                              'support data sets that have a string-based '
                              'dtype as it uses scikit-learn implementation '
                              'of decision trees.')
        structured_array_error = ('The TabularBlimeyTree explainer does not '
                                  'support structured data arrays as it uses '
                                  'scikit-learn implementation of decision '
                                  'trees.')

        with pytest.raises(TypeError) as exin:
            ftps.TabularBlimeyTree(futt.NUMERICAL_STRUCT_ARRAY,
                                   self.numerical_struct_array_classifier)
        assert str(exin.value) == structured_array_error

        categorical_array_classifier = fum.KNN(k=3)
        categorical_array_classifier.fit(futt.CATEGORICAL_NP_ARRAY,
                                         futt.LABELS)
        with pytest.raises(TypeError) as exin:
            ftps.TabularBlimeyTree(futt.CATEGORICAL_NP_ARRAY,
                                   categorical_array_classifier)
        assert str(exin.value) == string_array_error

        # Assert indices
        assert (  # yapf: disbale
            self.numerical_np_tabular_blimey.numerical_indices == [0, 1, 2, 3])
        assert self.numerical_np_tabular_blimey.categorical_indices == []
        assert isinstance(self.numerical_np_tabular_blimey.augmenter,
                          fuda.Mixup)

        assert (  # yapf: disbale
            self.numerical_np_cat_tabular_blimey.numerical_indices == [2, 3])
        assert (  # yapf: disbale
            self.numerical_np_cat_tabular_blimey.categorical_indices == [0, 1])
        assert isinstance(self.numerical_np_tabular_blimey.augmenter,
                          fuda.Mixup)

    def test_explain_instance_input_is_valid(self):
        """
        Tests the ``_explain_instance_input_is_valid`` method.

        Tests the :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyTree._explain_instance_input_is_valid` method.
        """
        explained_class_type = ('The explained_class parameter must be either '
                                'None, a string or an integer.')
        one_vs_rest_type = ('The one_vs_rest parameter must be either '
                            'None or a boolean.')
        samples_number_type = ('The samples_number parameter must be an '
                               'integer.')
        samples_number_value = ('The samples_number parameter must be a '
                                'positive (larger than 0) integer.')
        maximum_depth_type = 'The maximum_depth parameter must be an integer.'
        maximum_depth_value = ('The maximum_depth parameter must be a '
                               'positive (larger than 0) integer.')
        return_models_type = 'The return_models parameter must be a boolean.'

        array = futt.NUMERICAL_NP_ARRAY[0]

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7.0, None, None, None, None)
        assert str(exin.value) == explained_class_type

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7, 'False', None, None, None)
        assert str(exin.value) == one_vs_rest_type
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7, None, None, None, None)
        assert str(exin.value) == one_vs_rest_type

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, None, False, None, None, None)
        assert str(exin.value) == samples_number_type
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, None, False, 'a', None, None)
        assert str(exin.value) == samples_number_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 'a', True, 0, None, None)
        assert str(exin.value) == samples_number_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7, False, 1, None, None)
        assert str(exin.value) == maximum_depth_type
        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7, False, 1, 'a', None)
        assert str(exin.value) == maximum_depth_type
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, None, True, 1, 0, None)
        assert str(exin.value) == maximum_depth_value

        with pytest.raises(TypeError) as exin:
            self.numerical_np_tabular_blimey._explain_instance_input_is_valid(
                array, 7, False, 1, 1, None)
        assert str(exin.value) == return_models_type

        # All good
        assert self.numerical_np_tabular_blimey.\
            _explain_instance_input_is_valid(array, '3', False, 10, 3, False)
        assert self.numerical_np_cat_tabular_blimey.\
            _explain_instance_input_is_valid(array, 3, True, 10, 3, False)

    def test_get_local_model(self):
        """
        Tests the ``_get_local_model`` method.

        Tests the :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyTree._get_local_model` method.
        """
        fatf.setup_random_seed()

        one_runtime_error = ('A surrogate for the *{}* class (class index: '
                             '{}; class name: {}) could not be fitted as none '
                             'of the sampled data points were predicted (by '
                             'the black-box model) as this particular class.')
        multi_runtime_error = ('A surrogate model (classifier) could not be '
                               'fitted as the (black-box) predictions for the '
                               'sampled data are of a single class: *{}*.')

        sampled_data = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]])
        sampled_data_predictions_prob = np.array([[0.90, 0.08, 0.02],
                                                  [0.05, 0.80, 0.15],
                                                  [0.20, 0.10, 0.70]])
        sampled_data_predictions_noprob_cat = np.array(['a', 'b', 'c'])
        sampled_data_predictions_noprob_err_cat = np.array(['b', 'b', 'b'])

        tree_thresholds = np.array([0.5, -2, -2])

        # Test a probabilistic model
        model = self.numerical_np_tabular_blimey._get_local_model(
            sampled_data, sampled_data_predictions_prob, 1, True, 1)
        assert np.array_equal(model.tree_.threshold, tree_thresholds)

        # Test a non-probabilistic model
        # ...one-vs-rest
        with pytest.raises(RuntimeError) as exin:
            self.numerical_np_tabular_blimey_noprob._get_local_model(
                sampled_data, sampled_data_predictions_noprob_err_cat, 2, True,
                1)
        assert str(exin.value) == one_runtime_error.format('c', 2, 'class 2')

        model = self.numerical_np_tabular_blimey_noprob._get_local_model(
            sampled_data, sampled_data_predictions_noprob_cat, 1, True, 1)
        assert np.array_equal(model.tree_.threshold, tree_thresholds)

        # ...multi-class
        with pytest.raises(RuntimeError) as exin:
            self.numerical_np_tabular_blimey_noprob._get_local_model(
                sampled_data, sampled_data_predictions_noprob_err_cat, 2,
                False, 1)
        assert str(exin.value) == multi_runtime_error.format('b')

        model = self.numerical_np_tabular_blimey_noprob._get_local_model(
            sampled_data, sampled_data_predictions_noprob_cat, 1, False, 1)
        assert np.array_equal(model.tree_.threshold, tree_thresholds)

    def test_explain_instance(self, caplog):
        """
        Tests the ``explain_instance`` method.

        Tests the :func:`fatf.transparency.predictions.surrogate_explainers.\
TabularBlimeyTree.explain_instance` method.
        """
        fatf.setup_random_seed()

        probabilistic_user_warning = (
            'The one_vs_rest parameter cannot be set to False for '
            'probabilistic models, since a regression tree is fitted '
            'to the probabilities of each (or just the selected) '
            'class. This parameter setting will be ignored. Please '
            'see the documentation of the TabularBlimeyTree class for '
            'more details.')
        nonprobabilistic_user_warning = (
            'Choosing a class to explain (via the explained_class '
            'parameter) when the one_vs_rest parameter is set to '
            'False is not required as a single multi-class '
            'classification tree will be learnt regardless of the '
            'explained_class parameter setting.')

        nonprobabilistic_debug_log_prediction = (
            'Using the explained_class parameter as a *unique prediction* '
            'name for a classifier.')
        nonprobabilistic_debug_log_class = (
            'Using the explained_class parameter as a *class name* for a '
            'classifier.')
        nonprobabilistic_debug_log_int = (
            'Using the explained_class parameter as a class index for a '
            'classifier.')

        nonprobabilistic_info_log = (
            'A multi-class surrogate for a non-probabilistic '
            'black-box model is the same for all the possible '
            'classes, therefore a single model will be '
            'trained and used for explaining all of the '
            'classes.')

        probabilistic_value_error_str = ('The *{}* explained class name was '
                                         'not recognised. The following '
                                         'class names are allowed: {}.')
        probabilistic_value_error_int = ('The explained class index is out of '
                                         'the allowed range: 0 to {} (there '
                                         'are {} classes altogether).')

        nonprobabilistic_value_error_int = (
            'The explained_class parameter was not recognised as one of the '
            'possible class names and when treated as a class name index, '
            'it is outside of the allowed range: 0 to {} (there are {} '
            'classes altogether).')
        nonprobabilistic_value_error = (
            'The explained_class was not recognised. The following '
            'predictions: {}; and class names are allowed: {}. Alternatively, '
            'this parameter can be used to indicate the index of the class '
            '(from the list above) to be explained.')

        mixup_warning = (
            'Since the ground truth vector was not provided while '
            'initialising the Mixup class it is not possible to get a '
            'stratified sample of data points. Instead, Mixup will choose '
            'data points at random, which is equivalent to assuming that the '
            'class distribution is balanced.')

        numerical_np_explanation = {
            'class 0': {
                'feature 0': 0.552,
                'feature 1': 0.448,
                'feature 2': 0.0,
                'feature 3': 0.0
            },
            'class 1': {
                'feature 0': 0.0,
                'feature 1': 1.0,
                'feature 2': 0.0,
                'feature 3': 0.0
            },
            'class 2': {
                'feature 0': 0.564,
                'feature 1': 0.436,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        numerical_np_explanation_reg = {
            'feature 0': 0.432,
            'feature 1': 0.568,
            'feature 2': 0.0,
            'feature 3': 0.0
        }
        numerical_np_cat_explanation = {
            'class 0': {
                'feature 0': 0.0,
                'feature 1': 0.253,
                'feature 2': 0.288,
                'feature 3': 0.458
            },
            'class 1': {
                'feature 0': 0.0,
                'feature 1': 0.428,
                'feature 2': 0.485,
                'feature 3': 0.087
            },
            'class 2': {
                'feature 0': 0.072,
                'feature 1': 0.067,
                'feature 2': 0.861,
                'feature 3': 0.0
            }
        }
        iris_explanation = {
            'setosa': {
                'petal length (cm)': 0.980,
                'petal width (cm)': 0.010,
                'sepal length (cm)': 0.010,
                'sepal width (cm)': 0.0
            },
            'versicolor': {
                'petal length (cm)': 0.782,
                'petal width (cm)': 0.218,
                'sepal length (cm)': 0.0,
                'sepal width (cm)': 0.0
            },
            'virginica': {
                'petal length (cm)': 0.109,
                'petal width (cm)': 0.667,
                'sepal length (cm)': 0.028,
                'sepal width (cm)': 0.196
            }
        }

        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == 'INFO'
        assert caplog.records[0].getMessage().startswith('Seeding RNGs ')
        assert caplog.records[1].levelname == 'INFO'
        assert caplog.records[1].getMessage() == 'Seeding RNGs with 42.'

        # Probabilistic surrogates
        # ...correct
        explanation = self.numerical_np_tabular_blimey.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0], samples_number=50, maximum_depth=3)
        assert futt.is_explanation_equal_dict(
            numerical_np_explanation, explanation, atol=1e-3)

        explanation = self.numerical_np_cat_tabular_blimey.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0], samples_number=50, maximum_depth=3)
        assert futt.is_explanation_equal_dict(
            numerical_np_cat_explanation, explanation, atol=1e-3)

        explanation = self.iris_blimey.explain_instance(
            IRIS_DATASET['data'][0], samples_number=50, maximum_depth=3)
        assert futt.is_explanation_equal_dict(
            iris_explanation, explanation, atol=1e-3)

        # ...incorrect explained_class string
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                samples_number=5,
                maximum_depth=3,
                explained_class='class 3')
        assert str(exin.value) == probabilistic_value_error_str.format(
            'class 3', ['class 0', 'class 1', 'class 2'])

        # ...incorrect explained_class integer
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                samples_number=5,
                maximum_depth=3,
                explained_class=3)
        assert str(exin.value) == probabilistic_value_error_int.format(2, 3)

        explanation_ = {
            'class 2': {
                'feature 0': 0.588,
                'feature 1': 0.412,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        keys_ = ['class 2']
        thresholds_ = np.array([1.004, 0.472, -2, -2, -2])
        # ...user-specified one_vs_rest=False
        # ...explain class integer
        # ...return models
        with pytest.warns(UserWarning) as warning:
            return_ = self.numerical_np_tabular_blimey.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                samples_number=50,
                maximum_depth=2,
                one_vs_rest=False,
                explained_class=2,
                return_models=True)
        explanation, models = return_
        assert len(warning) == 1
        assert str(warning[0].message) == probabilistic_user_warning
        assert futt.is_explanation_equal_dict(
            explanation_, explanation, atol=1e-3)
        assert list(models.keys()) == keys_
        assert np.allclose(
            models[keys_[0]].tree_.threshold, thresholds_, atol=1e-3)

        # ...explain class string
        explanation_ = {
            'class 2': {
                'feature 0': 0.533,
                'feature 1': 0.467,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        explanation = self.numerical_np_tabular_blimey.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=2,
            explained_class='class 2')
        assert futt.is_explanation_equal_dict(
            explanation_, explanation, atol=1e-3)

        assert len(caplog.records) == 2
        # Non-probabilistic surrogates
        # ...multi-class
        # ...with selected class to be explained warning
        # ...explained_class taken from unique_predictions
        with pytest.warns(UserWarning) as warning:
            exp = self.numerical_np_tabular_blimey_noprob.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                samples_number=50,
                maximum_depth=2,
                one_vs_rest=False,
                explained_class='c')
        assert len(warning) == 1
        assert str(warning[0].message) == nonprobabilistic_user_warning
        #
        assert len(caplog.records) == 3
        assert caplog.records[2].levelname == 'DEBUG'
        assert (caplog.records[2].getMessage()  # yapf: disable
                == nonprobabilistic_debug_log_prediction)
        #
        exp_ = {
            'class 2': {
                'feature 0': 0.650,
                'feature 1': 0.350,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        assert futt.is_explanation_equal_dict(exp_, exp, atol=1e-3)

        # ...one-vs-rest
        # ...explained_class taken from class_names
        exp_ = {
            'class 2': {
                'feature 0': 0.344,
                'feature 1': 0.656,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        exp = self.numerical_np_tabular_blimey_noprob.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=2,
            one_vs_rest=True,
            explained_class='class 2')
        assert len(caplog.records) == 4
        assert caplog.records[3].levelname == 'DEBUG'
        assert (
            caplog.records[3].getMessage() == nonprobabilistic_debug_log_class)
        assert futt.is_explanation_equal_dict(exp_, exp, atol=1e-3)

        # ...one-vs-rest
        # ...explained_class given as an index
        # ...
        # ...correct index
        exp_ = {
            'class 2': {
                'feature 0': 0.605,
                'feature 1': 0.395,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        exp = self.numerical_np_tabular_blimey_noprob.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=2,
            one_vs_rest=True,
            explained_class=2)
        assert len(caplog.records) == 5
        assert caplog.records[4].levelname == 'DEBUG'
        assert (
            caplog.records[4].getMessage() == nonprobabilistic_debug_log_int)
        assert futt.is_explanation_equal_dict(exp_, exp, atol=1e-3)
        # ...incorrect index
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey_noprob.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                one_vs_rest=True,
                explained_class=3)
        assert str(exin.value) == nonprobabilistic_value_error_int.format(2, 3)

        # ...incorrect explained_class
        with pytest.raises(ValueError) as exin:
            self.numerical_np_tabular_blimey_noprob.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0],
                one_vs_rest=True,
                explained_class='incorrect string index')
        assert str(exin.value) == nonprobabilistic_value_error.format(
            ['a', 'b', 'c'], ['class 0', 'class 1', 'class 2'])

        # ...multi-class (one_vs_rest=False)
        # ...selected_class is None -- all classes
        # ...multi-class info logged -- the same model for every class
        assert len(caplog.records) == 5
        exp = self.numerical_np_tabular_blimey_noprob.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=2,
            one_vs_rest=False)
        assert len(caplog.records) == 6
        assert caplog.records[5].levelname == 'INFO'
        assert caplog.records[5].getMessage() == nonprobabilistic_info_log
        #
        # Check whether all the explanations (i.e., models) are the same.
        exp_uni = {
            'feature 0': 0.448,
            'feature 1': 0.552,
            'feature 2': 0.0,
            'feature 3': 0.0
        }
        exp_ = {k: exp_uni for k in exp}
        assert futt.is_explanation_equal_dict(exp_, exp, atol=1e-3)

        # Regression
        with pytest.warns(UserWarning) as warning:
            exp = self.numerical_np_tabular_blimey_reg.explain_instance(
                futt.NUMERICAL_NP_ARRAY[0], samples_number=50, maximum_depth=3)
        assert len(warning) == 1
        assert str(warning[0].message) == mixup_warning
        assert futt.is_explanation_equal_dict(
            {'': numerical_np_explanation_reg}, {'': exp}, atol=1e-3)

        assert len(caplog.records) == 6

        # ...explain class string first indexed class
        explanation_ = {
            'class 0': {
                'feature 0': 0.485,
                'feature 1': 0.515,
                'feature 2': 0.0,
                'feature 3': 0.0
            }
        }
        explanation = self.numerical_np_tabular_blimey.explain_instance(
            futt.NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            maximum_depth=2,
            explained_class=0)
        assert futt.is_explanation_equal_dict(
            explanation_, explanation, atol=1e-3)
