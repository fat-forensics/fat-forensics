import numpy as np
import pytest

import fatf.core.model as fcm
from fatf.exceptions import (
    CustomValueError,
    MissingImplementationException,
    PrefittedModelException,
    UnfittedModelException,
    IncorrectShapeException
    )

class TestModel(object):
    def test_model(self):
        error_message = ('Can\'t instantiate abstract class Model with abstract'
                         ' methods __init__, fit, predict')
        with pytest.raises(TypeError) as exception_info:
            model = fcm.Model()
        assert str(exception_info.value) == error_message

    def test_predict_proba(self):
        with pytest.raises(MissingImplementationException):
            fcm.Model.predict_proba(object, np.ndarray((0,)))


class TestKNN(object):
    unfittedmodelexception_message = 'This model has not been fitted yet.'
    prefittedmodelexception_message = 'This model has already been fitted.'
    incorrectsizeXexception_message = 'X must be 2-D array.'
    fitemptyXexception_message = 'Cannot fit model to empty array.'
    wrongnumberlabelexception_message = ('Number of samples in X must be '
                                        'same as number of labels in y.')
    X = np.array([
            [0,0],
            [1,1],
            [-1,1],
            [-1,-1],
            [1,-1],
            [2,2],
        ])
    X_struct = np.array(
            [(0, 0),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
            (2, 2)], dtype=[('a', 'f'), ('b', 'f')])

    X_mix = np.array([
            ('a', 0),
            ('b', 1),
            ('c', 1),
            ('c', -1),
            ('a', -1),
            ('d', 2)], dtype=[('a', '<U6'), ('b', 'f')])
    X_n = 6
    y = np.array([0,1,0,0,0,1])
    unique_y = np.array([0,1])
    unique_y_counts = np.array([4,2])

    X_test = np.array([
            [-.5, -.5],
            [4, 4],
            [0, 2]
        ])
    X_test_struct = np.array([
            (-.5, -.5),
            (4, 4),
            (0, 2)], dtype=[('a', 'f'), ('b', 'f')])
    X_test_mix = np.array([
            ('f', -.5),
            ('e', 4),
            ('a', 2)], dtype=[('a', '<U6'), ('b', 'f')])
    X_3d = np.ones((6, 2, 2))

    def _test_unfitted_internals(self, knn_clf):
        assert np.equal(knn_clf._X, np.ndarray((0,))).all()
        assert np.equal(knn_clf._y, np.ndarray((0,))).all()
        assert np.equal(knn_clf._unique_y, np.ndarray((0,))).all()
        assert np.equal(knn_clf._unique_y_counts, np.ndarray((0,))).all()
        assert knn_clf._X_n == int()
        assert knn_clf._is_fitted is False

    def _test_fitted_internals(self, knn_clf):
        assert np.equal(knn_clf._X, self.X).all()
        assert np.equal(knn_clf._y, self.y).all()
        assert knn_clf._X_n == self.X_n
        assert np.equal(knn_clf._unique_y, self.unique_y).all()
        assert np.equal(knn_clf._unique_y_counts, self.unique_y_counts).all()
        assert knn_clf._is_fitted

    def test_knn(self):
        clf = fcm.FAT_KNN()
        assert clf._k == 3
        self._test_unfitted_internals(clf)

        clf = fcm.FAT_KNN(k=8)
        assert clf._k == 8
        self._test_unfitted_internals(clf)

        with pytest.raises(CustomValueError) as exception_info:
            clf = fcm.FAT_KNN(k=.8)
        assert 'integer' in str(exception_info.value)

        with pytest.raises(CustomValueError) as exception_info:
            clf = fcm.FAT_KNN(k=-5.5)
        assert 'positive' in str(exception_info.value)

    def test_fit(self):
        k = 2

        clf = fcm.FAT_KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(IncorrectShapeException) as exception_info:
            clf.fit(self.X_3d, self.y)
        assert self.incorrectsizeXexception_message == str(exception_info.value)

        with pytest.raises(IncorrectShapeException) as exception_info:
            clf.fit(self.X[0:3], self.y)
        assert self.wrongnumberlabelexception_message == str(exception_info.value)

        with pytest.raises(CustomValueError) as exception_info:
            clf.fit(np.array([]), self.y)
        assert self.fitemptyXexception_message == str(exception_info.value)

        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        clf.clear()
        clf.fit(self.X_struct, self.y)

        clf.clear()
        clf.fit(self.X_mix, self.y)

        with pytest.raises(PrefittedModelException) as exception_info:
            clf.fit(self.X, self.y)
        assert self.prefittedmodelexception_message == str(exception_info.value)

    def test_clear(self):
        k = 2

        clf = fcm.FAT_KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(UnfittedModelException) as exception_info:
            clf.clear()
        assert self.unfittedmodelexception_message == str(exception_info.value)

        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        clf.clear()
        self._test_unfitted_internals(clf)
        assert clf._k == k

    def test_predict(self):
        k = 2
        clf = fcm.FAT_KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(UnfittedModelException) as exception_info:
            clf.predict(self.X_test)
        assert self.unfittedmodelexception_message == str(exception_info.value)

        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        y_test_labels = np.array([0, 1, 0])
        y_test_predictions = clf.predict(self.X_test)
        assert np.equal(y_test_labels, y_test_predictions).all()

        clf.clear()
        clf.fit(self.X_struct, self.y)
        y_test_predictions_struct = clf.predict(self.X_test_struct)
        assert np.equal(y_test_labels, y_test_predictions_struct).all()
    
        clf.clear()
        y_test_mix_labels = np.array([0, 1, 1])
        clf.fit(self.X_mix, self.y)
        y_test_predictions_mix = clf.predict(self.X_test_mix)
        assert np.equal(y_test_mix_labels, y_test_predictions_mix).all()

        # k is larger than the number of training data points -- majority label
        k = self.X_n + 5
        clf = fcm.FAT_KNN(k=k)
        clf.fit(self.X, self.y)

        y_test_labels = np.array([0, 0, 0])
        y_test_predictions = clf.predict(self.X_test)
        assert np.equal(y_test_labels, y_test_predictions).all()

        empty_predictions = clf.predict(np.array([]))
        assert np.equal(empty_predictions, np.array([])).all()

        with pytest.raises(IncorrectShapeException) as exception_info:
            clf.predict(self.X_3d)
        assert self.incorrectsizeXexception_message == str(exception_info.value)

    def test_predict_proba(self):
        k = 2

        clf = fcm.FAT_KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(UnfittedModelException) as exception_info:
            clf.predict(self.X_test)
        assert self.unfittedmodelexception_message == str(exception_info.value)
        
        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        y_test_true_probabilities = np.array([[1., 0,], [0., 1.], [0.5, 0.5]])
        y_test_probabilities = clf.predict_proba(self.X_test)
        assert np.equal(y_test_true_probabilities, y_test_probabilities).all()

        clf.clear()
        clf.fit(self.X_struct, self.y)
        y_test_probabilities_struct = clf.predict_proba(self.X_test_struct)
        assert np.equal(y_test_true_probabilities, 
                        y_test_probabilities_struct).all()

        y_test_true_probabilities_mix = np.array([[1., 0], [0., 1.], [0., 1.]])
        clf.clear()
        clf.fit(self.X_mix, self.y)
        y_test_probabilities_mix = clf.predict_proba(self.X_test_mix)
        assert np.equal(y_test_true_probabilities, 
                        y_test_probabilities_struct).all()
        empty_predictions = clf.predict_proba(np.array([]))
        assert np.equal(empty_predictions, np.array([])).all()

        with pytest.raises(IncorrectShapeException) as exception_info:
            clf.predict_proba(self.X_3d)
        assert self.incorrectsizeXexception_message == str(exception_info.value)
