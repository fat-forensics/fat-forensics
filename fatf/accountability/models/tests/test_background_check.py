"""
Tests implementations of models accountability measures.
"""
# Author: Miquel Perello Nieto <miquel.perellonieto@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.utils.data.datasets import load_iris
import fatf.utils.models as fatf_models
from fatf.accountability.models.background_check import BackgroundCheck
from fatf.accountability.models.background_check import update_posterior

from fatf.utils.data.density import GaussianRelativeDensityEstimator

def test_background_check_pipeline():
    """
    Checks that a full training and testing pipeline runs.
    """
    # Load data
    iris_data_dict = load_iris()
    X = iris_data_dict['data']
    y = iris_data_dict['target'].astype(int)
    n_classes = len(iris_data_dict['target_names'])

    # Train probabilistic classifier
    clf = fatf_models.KNN()
    clf.fit(X, y)

    # Train a relative density estimator
    rde = GaussianRelativeDensityEstimator()
    rde.fit(X)

    # Create Background Check
    bc = BackgroundCheck(clf, rde)

    # Test Background Check
    class_bg_posteriors = bc.predict_proba(X, mu0=1.0, mu1=0.0)

    # Same number of samples
    assert(class_bg_posteriors.shape[0] == X.shape[0])
    # One additional class
    assert(class_bg_posteriors.shape[1] == n_classes + 1)


