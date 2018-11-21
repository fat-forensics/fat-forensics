"""
Tests for LIME wrapper
Author: Alex Hepburn <ah13558@bristol.ac.uk>
License: new BSD
"""
# DO I NEED TO JUST MAKE SURE THAT IT RUNS SINCE ITS USING WRAPPER
# DO I NEED TO TEST LAYOUT OF FIGURES

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from fatf.analyse.lime import Lime
from fatf.dataset import Dataset

def test_plot():
    X = np.random.rand(10, 4)
    header = ['height', 'width', 'length', 'group']
    output = [1, 2, 2, 0, 1, 1, 0, 1, 2, 0]
    class_names = ['excellent', 'good' ,'bad']
    predictor = LogisticRegression()
    predictor.fit(X, output)
    train = Dataset(X, output, header, class_names)
    l = Lime(train, predictor, categorical=['width'])
    l.explain_instance(np.array([100,2,5,2]))


if __name__ == '__main__':
    test_plot()
