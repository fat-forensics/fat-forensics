"""
Testing feature importance that is a wrapper for skater package
Author: Alex Hepburn <ah13558@bristol.ac.uk>
License: new BSD
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from fatf.dataset import Dataset
from fatf.analyse.feature_importance import *

def test_compute_feature_importance():
    X = np.random.rand(10, 4)
    header = ['height', 'width', 'length', 'group']
    output = [1, 2, 2, 0, 1, 1, 0, 1, 2, 0]
    class_names = ['excellent', 'good' ,'bad']
    predictor = LogisticRegression()
    predictor.fit(X, output)
    train = Dataset(X, output, header, class_names)
    importance = compute_feature_importance(train, predictor, mode='serires')
    print(importance)

if __name__ == '__main__':
    test_compute_feature_importance()
