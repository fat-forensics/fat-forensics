import numpy as np
import pytest

import fatf.core.model as fcm
from fatf.exceptions import (
    CustomValueError,
    MissingImplementationException,
    PrefittedModelException,
    UnfittedModelException
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

    X = np.array([
            [0,0],
            [1,1],
            [-1,1],
            [-1,-1],
            [1,-1],
            [2,2],
        ])
    X_n = 6
    y = np.array([0,1,0,0,0,1])
    unique_y = np.array([0,1])
    unique_y_counts = np.array([4,2])

    X_test = np.array([
            [-.5, -.5],
            [4, 4],
            [0, 2]
        ])

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
        clf = fcm.KNN()
        assert clf._k == 3
        self._test_unfitted_internals(clf)

        clf = fcm.KNN(k=8)
        assert clf._k == 8
        self._test_unfitted_internals(clf)

        with pytest.raises(CustomValueError) as exception_info:
            clf = fcm.KNN(k=.8)
        assert 'integer' in str(exception_info.value)

        with pytest.raises(CustomValueError) as exception_info:
            clf = fcm.KNN(k=-5.5)
        assert 'positive' in str(exception_info.value)

    def test_fit(self):
        k = 2

        clf = fcm.KNN(k=k)
        self._test_unfitted_internals(clf)

        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        with pytest.raises(PrefittedModelException) as exception_info:
            clf.fit(self.X, self.y)
        assert self.prefittedmodelexception_message == str(exception_info.value)

    def test_clear(self):
        k = 2

        clf = fcm.KNN(k=k)
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
        clf = fcm.KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(UnfittedModelException) as exception_info:
            clf.predict(self.X_test)
        assert self.unfittedmodelexception_message == str(exception_info.value)

        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf)

        y_test_labels = np.array([0, 1, 0])
        y_test_predictions = clf.predict(self.X_test)
        assert np.equal(y_test_labels, y_test_predictions).all()

        # k is larger than the number of training data points -- majority label
        k = self.X_n + 5
        clf = fcm.KNN(k=k)
        clf.fit(self.X, self.y)

        y_test_labels = np.array([0, 0, 0])
        y_test_predictions = clf.predict(self.X_test)
        assert np.equal(y_test_labels, y_test_predictions).all()

    def test_predict_proba(self):
        k = 2

        clf = fcm.KNN(k=k)
        self._test_unfitted_internals(clf)

        with pytest.raises(MissingImplementationException) as exception_info:
            clf.predict_proba(self.X_test)
