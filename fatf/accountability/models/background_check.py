import numpy as np


def background_posterior(relative_densities, mu0, mu1, density_range=(0., 1.)):

    if density_range is None:
        density_range = (0, 1)

    # assuming that the `relative_densities` is a numpy array
    if ((relative_densities < density_range[0]).any() or
        (relative_densities > density_range[1]).any()):
        raise ValueError('please specify the correct range {}'.format(denisty_range))

    p_x_and_b = relative_densities * mu1 + (1.0 - relative_densities) * mu0

    bg_fg_posteriors = np.zeros((relative_densities.shape[0], 2))
    bg_fg_posteriors[:, 0] = p_x_and_b / (p_x_and_b + relative_densities)
    bg_fg_posteriors[:, 1] = 1.0 - bg_fg_posteriors[:, 0]

    return bg_fg_posteriors


def update_posterior(bg_fg_posteriors, class_posteriors):
    class_posteriors = class_posteriors * bg_fg_posteriors[:, 1].reshape(-1, 1)
    updated_posteriors = np.hstack((class_posteriors,
                                    bg_fg_posteriors[:, 0].reshape(-1, 1)))
    return updated_posteriors


class BackgroundCheck(object):
    def __init__(self, clf, rde):
        '''
            clf: pretrained probabilistic classifier
                Per each samples always gives K probabilities that sum to one,
                where K is the number of classes.
                clf requires function predict_proba
            rde: pretrained relative density estimator
                Per each sample always gives one score between 0 and 1
        '''
        self._clf = clf
        self._rde = rde


    def predict_proba(self, X, mu0=1.0, mu1=0.0):
        """Performs background check on the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.
            mu0 (float):

        Returns:
            posteriors (array-like, shape = [n_samples, 2]): posterior
            probabilities for background (column 0) and foreground (column 1).

        """
        relative_densities = self._rde.score(X)

        bg_fg_posteriors = background_posterior(relative_densities, mu0, mu1)

        class_posteriors = self._clf.predict_proba(X)

        class_bg_posteriors = update_posterior(bg_fg_posteriors,
                                               class_posteriors)
        return class_bg_posteriors
