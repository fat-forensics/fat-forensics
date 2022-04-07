"""
Tets the :mod:`fatf.transparency.predictions.surrogate_image_explainers`
module.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import importlib
import pytest
import sys

import numpy as np

from fatf.exceptions import IncompatibleModelError

import fatf

import fatf.utils.models as fum
import fatf.utils.testing.imports as futi

try:
    import skimage
    import sklearn
    import PIL
except ImportError:  # pragma: no cover
    pytest.skip(
        'Skipping tests of surrogate image explainers -- scikit-image, '
        'scikit-learn or Pillow is not installed.',
        allow_module_level=True)
else:
    del skimage
    del sklearn
    del PIL

import fatf.transparency.predictions.surrogate_image_explainers as ftps
import fatf.utils.data.segmentation as fuds
import fatf.utils.data.occlusion as fudo

ONES = np.ones(shape=(2, 2, 3), dtype=np.uint8)
ARRAY_IMAGE_3D_0 = np.zeros(shape=(2, 2, 3), dtype=np.uint8)
ARRAY_IMAGE_3D_1 = np.full((2, 2, 3), 255, dtype=np.uint8)


def test_imports():
    """
    Tests importing the
    :mod:`fatf.transparency.predictions.surrogate_image_explainers` module.
    """
    # Present
    # scikit-learn
    mod = 'fatf.transparency.predictions.surrogate_image_explainers'
    assert mod in sys.modules
    with futi.module_import_tester('sklearn', when_missing=False):
        importlib.reload(
            fatf.transparency.predictions.surrogate_image_explainers)

    # Missing PIL
    import_msg = (
        'The ImageBlimeyLime surrogate image explainer requires scikit-learn, '
        'scikit-image and Pillow to be installed.\n\n'
        "No module named 'sklearn'")
    with futi.module_import_tester('sklearn', when_missing=True):
        with pytest.raises(ImportError) as exin:
            importlib.reload(
                fatf.transparency.predictions.surrogate_image_explainers)
        assert str(exin.value) == import_msg
    assert mod in sys.modules


class TestImageBlimeyLime(object):
    """
    Tests the :class:`fatf.transparency.predictions.\
surrogate_image_explainers.ImageBlimeyLime` class.
    """

    class BadClassifier(fum.models.Model):
        def __init__(self):
            pass

        def fit(self, X, y):
            return X + y

        def predict(self, X, y):
            return X + y

    class BadProbabilisticClassifier(fum.models.Model):
        def __init__(self):
            pass

        def fit(self, X, y):
            return X + y

        def predict(self, X):
            return X

        def predict_proba(self, X, y):
            return X + y

    class KNN(fum.KNN):
        def predict(self, X):
            X_ = np.array([x.flatten() for x in X])
            return super().predict(X_)

        def predict_proba(self, X):
            X_ = np.array([x.flatten() for x in X])
            return super().predict_proba(X_)

    def test_blimey_class_init(self, caplog):
        """
        Tests the :class:`fatf.transparency.predictions.\
surrogate_image_explainers.ImageBlimeyLime` class init.
        """
        clf = self.BadClassifier()
        clf_ = self.BadProbabilisticClassifier()
        knn = self.KNN(k=1)
        knn.fit(
            np.array([ARRAY_IMAGE_3D_0.flatten(),
                      ARRAY_IMAGE_3D_1.flatten()]), np.array([0, 1]))

        wrn_msg = (
            'Model object characteristics are neither consistent with '
            'supervised nor unsupervised models.\n\n'
            '--> Unsupervised models <--\n'
            "The 'fit' method of the *BadClassifier* (model) class has "
            'incorrect number (2) of the required parameters. It needs to '
            'have exactly 1 required parameter(s). Try using optional '
            'parameters if you require more functionality.\n'
            "The 'predict' method of the *BadClassifier* (model) class has "
            'incorrect number (2) of the required parameters. It needs to '
            'have exactly 1 required parameter(s). Try using optional '
            'parameters if you require more functionality.\n\n'
            '--> Supervised models <--\n'
            "The 'predict' method of the *BadClassifier* (model) class has "
            'incorrect number (2) of the required parameters. It needs to '
            'have exactly 1 required parameter(s). Try using optional '
            'parameters if you require more functionality.')
        wrn_msg_ = (
            'Model object characteristics are neither consistent with '
            'supervised nor unsupervised models.\n\n'
            '--> Unsupervised models <--\n'
            "The 'fit' method of the *BadProbabilisticClassifier* (model) "
            'class has incorrect number (2) of the required parameters. '
            'It needs to have exactly 1 required parameter(s). Try using '
            'optional parameters if you require more functionality.\n\n'
            '--> Supervised models <--\n'
            "The 'predict_proba' method of the *BadProbabilisticClassifier* "
            '(model) class has incorrect number (2) of the required '
            'parameters. It needs to have exactly 1 required parameter(s). '
            'Try using optional parameters if you require more functionality.')

        log_1 = 'Building segmentation.'
        log_2 = 'Building occlusion.'
        assert len(caplog.records) == 0

        err = 'The as_probabilistic parameter must be a boolean.'
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, as_probabilistic='bool')
        assert str(exin.value) == err

        err = ('With as_probabilistic set to True the predictive model '
               'needs to be capable of outputting probabilities via '
               'a *predict_proba* method, which takes exactly one '
               'required parameter -- data to be predicted -- and '
               'outputs a 2-dimensional array with probabilities.')
        with pytest.warns(UserWarning) as warning:
            with pytest.raises(IncompatibleModelError) as exin:
                ftps.ImageBlimeyLime(
                    ARRAY_IMAGE_3D_1, clf_, as_probabilistic=True)
            assert str(exin.value) == err
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg_

        err = ('With as_probabilistic set to False the predictive model '
               'needs to be capable of outputting (class) predictions '
               'via a *predict* method, which takes exactly one required '
               'parameter -- data to be predicted -- and outputs a '
               '1-dimensional array with (class) predictions.')
        with pytest.warns(UserWarning) as warning:
            with pytest.raises(IncompatibleModelError) as exin:
                ftps.ImageBlimeyLime(
                    ARRAY_IMAGE_3D_1, clf, as_probabilistic=False)
            assert str(exin.value) == err
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg

        # Class names errors
        err = 'The class_names parameter must be a Python list or None.'
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(ARRAY_IMAGE_3D_1, knn, class_names='list')
        assert str(exin.value) == err

        err = 'The class_names list cannot be empty.'
        with pytest.raises(ValueError) as exin:
            ftps.ImageBlimeyLime(ARRAY_IMAGE_3D_1, knn, class_names=[])
        assert str(exin.value) == err

        err = 'The class_names list contains duplicated entries.'
        with pytest.raises(ValueError) as exin:
            ftps.ImageBlimeyLime(ARRAY_IMAGE_3D_1, knn, class_names=[1, 2, 1])
        assert str(exin.value) == err

        err = ('All elements of the class_names list must be strings or '
               'integers; *42* is not.')
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['24', 42, '82'])
        assert str(exin.value) == err
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=[42, 84, '42'])
        assert str(exin.value) == err
        err = ('All elements of the class_names list must be strings or '
               'integers; *42.0* is not.')
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=[84, 42., 24])
        assert str(exin.value) == err
        with pytest.raises(TypeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=[42., 24, 84])
        assert str(exin.value) == err

        err = ('The number of class names does not correspond to the shape of '
               'the model predictions.')
        with pytest.raises(RuntimeError) as exin:
            ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['ones', 'zeros', 'O'])
        assert str(exin.value) == err

        # Probabilistic -- attributes & logging
        assert len(caplog.records) == 0

        wrn_seg = ('The segmentation returned only **one** segment. Consider '
                   'tweaking the parameters to generate a reasonable '
                   'segmentation.')
        wrn_occ = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            blimey = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['ones', 'zeros'])
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ

        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == 'DEBUG'
        assert caplog.records[0].getMessage() == log_1
        assert caplog.records[1].levelname == 'DEBUG'
        assert caplog.records[1].getMessage() == log_2

        assert np.array_equal(blimey.image, ARRAY_IMAGE_3D_1)
        assert np.array_equal(blimey.image, blimey.segmentation_mask)
        assert isinstance(blimey.segmenter, fuds.QuickShift)
        assert isinstance(blimey.occluder, fudo.Occlusion)
        assert blimey.as_probabilistic
        assert blimey.predictive_model is knn
        assert (blimey.predictive_function
                == blimey.predictive_model.predict_proba)  # yapf: disable
        assert blimey.image_prediction == 1
        assert blimey.classes_number == 2
        assert np.array_equal(blimey.class_names, ['ones', 'zeros'])
        assert blimey.surrogate_data_sample is None
        assert blimey.surrogate_data_predictions is None
        assert blimey.similarities is None

        # Crisp -- attributes & logging
        # with segmentation mask & class names & merge list
        wrn_seg_setter = 'The segmentation has only **one** segment.'
        assert len(caplog.records) == 2
        with pytest.warns(UserWarning) as warning:
            blimey = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_0,
                knn,
                class_names=['ones', 'zeros'],
                as_probabilistic=False,
                segmentation_mask=ARRAY_IMAGE_3D_1,
                segments_merge_list=[1])
        assert len(warning) == 3
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_seg_setter
        assert str(warning[2].message) == wrn_occ

        assert len(caplog.records) == 4
        assert caplog.records[2].levelname == 'DEBUG'
        assert caplog.records[2].getMessage() == log_1
        assert caplog.records[3].levelname == 'DEBUG'
        assert caplog.records[3].getMessage() == log_2

        assert np.array_equal(blimey.image, ARRAY_IMAGE_3D_0)
        assert np.array_equal(blimey.segmentation_mask, ARRAY_IMAGE_3D_1)
        assert isinstance(blimey.segmenter, fuds.QuickShift)
        assert isinstance(blimey.occluder, fudo.Occlusion)
        assert not blimey.as_probabilistic
        assert blimey.predictive_model is knn
        assert blimey.predictive_function == blimey.predictive_model.predict
        assert blimey.image_prediction == 0
        assert blimey.classes_number == 2
        assert np.array_equal(blimey.class_names, ['ones', 'zeros'])
        assert blimey.surrogate_data_sample is None
        assert blimey.surrogate_data_predictions is None
        assert blimey.similarities is None

    def test_set_occlusion_colour(self):
        """
        Tests the :func:`fatf.transparency.predictions.\
surrogate_image_explainers.ImageBlimeyLime.set_occlusion_colour` method.
        """
        knn = self.KNN(k=1)
        knn.fit(
            np.array([ARRAY_IMAGE_3D_0.flatten(),
                      ARRAY_IMAGE_3D_1.flatten()]), np.array([0, 1]))

        wrn_seg = ('The segmentation returned only **one** segment. Consider '
                   'tweaking the parameters to generate a reasonable '
                   'segmentation.')
        wrn_occ = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            blimey = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['ones', 'zeros'])
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ

        assert np.array_equal(
            blimey.occluder._colouring_strategy(ONES),
            blimey.occluder._generate_colouring_strategy('mean')(ONES))

        blimey.set_occlusion_colour('black')
        assert np.array_equal(
            blimey.occluder._colouring_strategy(ONES),
            blimey.occluder._generate_colouring_strategy('black')(ONES))

    def test_explain_instance_errors(self):
        """
        Tests errors of the :func:`fatf.transparency.predictions.\
surrogate_image_explainers.ImageBlimeyLime.explain_instance` method.
        """
        knn = self.KNN(k=1)
        knn.fit(
            np.array([ARRAY_IMAGE_3D_0.flatten(),
                      ARRAY_IMAGE_3D_1.flatten()]), np.array([0, 1]))

        wrn_seg = ('The segmentation returned only **one** segment. Consider '
                   'tweaking the parameters to generate a reasonable '
                   'segmentation.')
        wrn_occ = 'The segmentation has only **one** segment.'
        with pytest.warns(UserWarning) as warning:
            blimey = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1,
                knn,
                class_names=['ones', 'zeros'],
                as_probabilistic=False)
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ
        with pytest.warns(UserWarning) as warning:
            blimey_ = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['ones', 'zeros'])
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ
        with pytest.warns(UserWarning) as warning:
            blimey__ = ftps.ImageBlimeyLime(ARRAY_IMAGE_3D_1, knn)
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ

        err = 'The return_model parameter should be a boolean.'
        with pytest.raises(TypeError) as exin:
            blimey_.explain_instance(return_model='bool')
        assert str(exin.value) == err

        err = 'The reuse_sample parameter should be a boolean.'
        with pytest.raises(TypeError) as exin:
            blimey_.explain_instance(reuse_sample='bool')
        assert str(exin.value) == err

        err = ('The explained_class parameter must be either of '
               'None, a string or an integer.')
        with pytest.raises(TypeError) as exin:
            blimey_.explain_instance(explained_class=['list'])
        assert str(exin.value) == err
        with pytest.raises(TypeError) as exin:
            blimey_.explain_instance(explained_class=4.2)
        assert str(exin.value) == err

        err = 'The explained class index is invalid.'
        with pytest.raises(IndexError) as exin:
            blimey_.explain_instance(explained_class=-1)
        assert str(exin.value) == err
        with pytest.raises(IndexError) as exin:
            blimey_.explain_instance(explained_class=2)
        assert str(exin.value) == err

        err = ('It is not possible to use a name for the explained '
               'class without initialising this explainer with a '
               'list of class names (the *class_names* parameter).')
        with pytest.raises(RuntimeError) as exin:
            blimey__.explain_instance(explained_class='twos')
        assert str(exin.value) == err

        err = ('The name of the explained class could not be '
               'found in the list of class names used to '
               'initialise this explainer (the *class_names* '
               'parameter).')
        with pytest.raises(IndexError) as exin:
            blimey_.explain_instance(explained_class='twos')
        assert str(exin.value) == err

        err = ('The name of the explained class could not be found '
               'in the list of class names used to initialise this '
               'explainer (the *class_names* parameter).')
        with pytest.raises(IndexError) as exin:
            blimey.explain_instance(explained_class='twos')
        assert str(exin.value) == err

        err = ('You need to explain an instance before '
               'being able to reuse its (random) sample.')
        with pytest.raises(RuntimeError) as exin:
            blimey_.explain_instance(reuse_sample=True)
        assert str(exin.value) == err

    def test_explain_instance(self, caplog):
        """
        Tests the :func:`fatf.transparency.predictions.\
surrogate_image_explainers.ImageBlimeyLime.explain_instance` method.
        """
        log_exp = ('Generating missing class names from the array of classes '
                   'output by the classifier using "class %s" pattern.')
        log_0_0 = 'Building segmentation.'
        log_0_1 = 'Building occlusion.'
        wrn_msg = ('None of the sampled data points were predicted by the '
                   'model with the explained class. The explanation may be '
                   'untrustworthy or the name of the explained class has '
                   'been missspelled!')
        log_1 = 'Reusing the sample.'
        log_2 = 'Generating a sample.'
        log_3 = 'Computing distances.'
        log_4 = 'Setting the distance to all-0 vectors to 1.'
        log_5 = 'Transforming distances into similarities.'
        log_6 = 'Reconstructing and predicting images.'
        log_7 = 'Fitting the surrogate.'

        knn = self.KNN(k=1)
        knn.fit(
            np.array([ARRAY_IMAGE_3D_0.flatten(),
                      ARRAY_IMAGE_3D_1.flatten()]), np.array([0, 1]))

        wrn_seg = ('The segmentation returned only **one** segment. Consider '
                   'tweaking the parameters to generate a reasonable '
                   'segmentation.')
        wrn_occ = 'The segmentation has only **one** segment.'

        assert len(caplog.records) == 0
        with pytest.warns(UserWarning) as warning:
            blimey = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, as_probabilistic=False)
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ
        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == 'DEBUG'
        assert caplog.records[0].getMessage() == log_0_0
        assert caplog.records[1].levelname == 'DEBUG'
        assert caplog.records[1].getMessage() == log_0_1

        assert len(caplog.records) == 2
        with pytest.warns(UserWarning) as warning:
            blimey_ = ftps.ImageBlimeyLime(
                ARRAY_IMAGE_3D_1, knn, class_names=['ones', 'zeros'])
        assert len(warning) == 2
        assert str(warning[0].message) == wrn_seg
        assert str(warning[1].message) == wrn_occ
        assert len(caplog.records) == 4
        assert caplog.records[2].levelname == 'DEBUG'
        assert caplog.records[2].getMessage() == log_0_0
        assert caplog.records[3].levelname == 'DEBUG'
        assert caplog.records[3].getMessage() == log_0_1

        # Generating sample for probabilistic -- no model return
        assert len(caplog.records) == 4
        fatf.setup_random_seed()
        assert len(caplog.records) == 6
        assert caplog.records[4].levelname == 'INFO'
        assert (caplog.records[4].getMessage()
                == 'Seeding RNGs using the system variable.')  # yapf: disable
        assert caplog.records[5].levelname == 'INFO'
        assert caplog.records[5].getMessage() == 'Seeding RNGs with 42.'

        assert len(caplog.records) == 6
        exp = blimey_.explain_instance(samples_number=200)
        assert 'Segment #1' in exp
        assert len(exp.keys()) == 1
        assert exp['Segment #1'] == 0

        assert len(caplog.records) == 12
        assert caplog.records[6].levelname == 'DEBUG'
        assert caplog.records[6].getMessage() == log_2
        assert caplog.records[7].levelname == 'DEBUG'
        assert caplog.records[7].getMessage() == log_3
        assert caplog.records[8].levelname == 'DEBUG'
        assert caplog.records[8].getMessage() == log_4
        assert caplog.records[9].levelname == 'DEBUG'
        assert caplog.records[9].getMessage() == log_5
        assert caplog.records[10].levelname == 'DEBUG'
        assert caplog.records[10].getMessage() == log_6
        assert caplog.records[11].levelname == 'DEBUG'
        assert caplog.records[11].getMessage() == log_7

        # Rusing sample
        assert len(caplog.records) == 12
        exp = blimey_.explain_instance(
            reuse_sample=True, explained_class='ones')
        assert 'Segment #1' in exp
        assert len(exp.keys()) == 1
        assert exp['Segment #1'] == 0

        assert len(caplog.records) == 14
        assert caplog.records[12].levelname == 'DEBUG'
        assert caplog.records[12].getMessage() == log_1
        assert caplog.records[13].levelname == 'DEBUG'
        assert caplog.records[13].getMessage() == log_7

        # Regenerate and compare sample
        sample_ = blimey_.surrogate_data_sample.copy()
        pred_ = blimey_.surrogate_data_predictions.copy()
        sim_ = blimey_.similarities.copy()
        assert len(caplog.records) == 14
        exp = blimey_.explain_instance()
        assert 'Segment #1' in exp
        assert len(exp.keys()) == 1
        assert exp['Segment #1'] == 0

        assert not np.array_equal(sample_, blimey_.surrogate_data_sample)
        assert not np.array_equal(pred_, blimey_.surrogate_data_predictions)
        assert not np.array_equal(sim_, blimey_.similarities)

        assert len(caplog.records) == 20
        assert caplog.records[14].levelname == 'DEBUG'
        assert caplog.records[14].getMessage() == log_2
        assert caplog.records[15].levelname == 'DEBUG'
        assert caplog.records[15].getMessage() == log_3
        assert caplog.records[16].levelname == 'DEBUG'
        assert caplog.records[16].getMessage() == log_4
        assert caplog.records[17].levelname == 'DEBUG'
        assert caplog.records[17].getMessage() == log_5
        assert caplog.records[18].levelname == 'DEBUG'
        assert caplog.records[18].getMessage() == log_6
        assert caplog.records[19].levelname == 'DEBUG'
        assert caplog.records[19].getMessage() == log_7

        # Generating sample for crisp -- with a custom colour and return model
        assert len(caplog.records) == 20
        exp, mdl = blimey.explain_instance(
            samples_number=200, colour='black', return_model=True)
        assert 'Segment #1' in exp
        assert len(exp.keys()) == 1
        assert pytest.approx(exp['Segment #1'], abs=1e-3) == 0.060
        assert pytest.approx(mdl.coef_[0, 0], abs=1e-3) == 0.060

        assert len(caplog.records) == 27
        assert caplog.records[20].levelname == 'DEBUG'
        assert caplog.records[20].getMessage() == log_2
        assert caplog.records[21].levelname == 'DEBUG'
        assert caplog.records[21].getMessage() == log_3
        assert caplog.records[22].levelname == 'DEBUG'
        assert caplog.records[22].getMessage() == log_4
        assert caplog.records[23].levelname == 'DEBUG'
        assert caplog.records[23].getMessage() == log_5
        assert caplog.records[24].levelname == 'DEBUG'
        assert caplog.records[24].getMessage() == log_6
        assert caplog.records[25].levelname == 'DEBUG'
        assert caplog.records[25].getMessage() == log_7
        assert caplog.records[26].levelname == 'INFO'
        assert caplog.records[26].getMessage() == log_exp

        # Rusing sample -- misspell class name & reuse sample
        sample_ = blimey.surrogate_data_sample.copy()
        pred_ = blimey.surrogate_data_predictions.copy()
        sim_ = blimey.similarities.copy()
        assert len(caplog.records) == 27
        with pytest.warns(UserWarning) as warning:
            exp = blimey.explain_instance(
                reuse_sample=True, explained_class=42)
        assert len(warning) == 1
        assert str(warning[0].message) == wrn_msg
        assert 'Segment #1' in exp
        assert len(exp.keys()) == 1
        assert exp['Segment #1'] == 0

        assert np.array_equal(sample_, blimey.surrogate_data_sample)
        assert np.array_equal(pred_, blimey.surrogate_data_predictions)
        assert np.array_equal(sim_, blimey.similarities)

        assert len(caplog.records) == 30
        assert caplog.records[27].levelname == 'DEBUG'
        assert caplog.records[27].getMessage() == log_1
        assert caplog.records[28].levelname == 'DEBUG'
        assert caplog.records[28].getMessage() == log_7
        assert caplog.records[29].levelname == 'INFO'
        assert caplog.records[29].getMessage() == log_exp
