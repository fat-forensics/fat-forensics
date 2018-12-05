import pytest
import numpy as np

from fatf.tests.predictor import KNN
from fatf.analyse.feature_importance import *

NUMERICAL_ARRAY = np.array([
    [0, 0.78, 0.67],
    [1, 0.98, 0.51],
    [2, 0.34, 0.14],
    [1, 0.235, 0.32],
    [0, 0.25, 0.12]
    ])

STRUCTURE_ARRAY = np.array([
    ('A', 23, ),
    ('B', 30, ),
    ('B', 58, ),
    ('A', 80, ),
    ('A', 18, )
    ], dtype=[('disease', 'S1'), ('age', '<i4'), ('weight', '')])

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

if __name__ == '__main__':
    knn = KNN(k=2)
    X = np.array([
        [1.2, 2, 3],
        [2.3, 3, 4],
        [10.3, 2, 4],
        [4.0, 3, 5]
    ], dtype=np.float32)
    Y = np.array([0, 0, 1, 1])
    #X = np.random.rand(100, 10)
    #Y = np.random.randint(0,2,100)
    knn.fit(X, Y)
    ret, values = individual_conditional_expectation(X, knn, 0)
    pd = partial_depedence(X, knn, 0, 0)
    #plot_ICE(ret, 'feature 0', values, 0, 'Good')
    imp = feature_importance(X, knn)
