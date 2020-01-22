import numpy as np
from scipy.special import expit
from sklearn.svm import OneClassSVM


def background_posterior(scores, mu0, mu1, delta, max_dens):
    # TODO maybe apply expit only on the necessary cases
    p_x_and_f = expit(scores + delta)
    # TODO look for other methods to clip the probabilities?
    q = np.clip(p_x_and_f / max_dens, 0.0, 1.0)
    p_x_and_b = q * mu1 + (1.0 - q) * mu0

    bg_fg_posteriors = np.zeros((len(scores), 2))
    bg_fg_posteriors[:, 0] = p_x_and_b / (p_x_and_b + q)
    bg_fg_posteriors[:, 1] = 1.0 - bg_fg_posteriors[:, 0]

    return bg_fg_posteriors


def update_posterior(bg_fg_posteriors, class_posteriors):
    class_posteriors = class_posteriors * bg_fg_posteriors[:, 1].reshape(-1, 1)

    return np.hstack((class_posteriors, bg_fg_posteriors[:, 0].reshape(-1, 1)))


class BackgroundCheck(object):
    def __init__(self, clf, density_estimator=OneClassSVM()):
        # Check that clf is probabilistic
        self._clf = clf
        self._density_estimator = density_estimator


    def fit(self, X):
        """Fits the density estimator to the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            Nothing.

        """
        self._density_estimator.fit(X)
        surrogate_densities = self._density_score(X)
        # Minimum density from the training set
        self._delta = - surrogate_densities.min()
        # Sets the 0.5 probability to the sample with minimum density
        relative_densities = expit(surrogate_densities + self._delta)
        self._max_dens  = relative_densities.max()


    def predict_proba(self, X, mu0=1.0, mu1=0.0):
        """Performs background check on the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.
            mu0 (float):

        Returns:
            posteriors (array-like, shape = [n_samples, 2]): posterior
            probabilities for background (column 0) and foreground (column 1).

        """
        surrogate_densities = self._density_score(X)
        bg_fg_posteriors = background_posterior(surrogate_densities, mu0, mu1,
                                                self._delta, self._max_dens)

        class_posteriors = self._clf.predict_proba(X)

        class_bg_posteriors = update_posterior(bg_fg_posteriors,
                                               class_posteriors)
        return class_bg_posteriors


    def _density_score(self, X):
        """Gets scores for the objects of X using different functions that
        depend on the estimator.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            (array-like, shape = [n_samples]): scores given by the estimator.

        """
        if 'score_samples' in dir(self._density_estimator):
            s = self._density_estimator.score_samples(X)
        elif 'score' in dir(self._density_estimator):
            s = self._density_estimator.score(X)
        elif 'decision_function' in dir(self._density_estimator):
            s = self._density_estimator.decision_function(X).reshape(-1)
        return s
