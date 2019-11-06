"""
Tests helper functions for validating transparency results.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import fatf.utils.testing.transparency as futt

# yapf: disable
DICT1 = {'a': [('a1', 0.1), ('a2', 0.9)],
         'b': [('b1', 0.55), ('b2', 0.2222)],
         'c': [('c', 7)]}
DICT2 = {'a': [('a2', 0.8), ('a1', 0.0)],
         'b': [('b1', 0.5), ('b2', 0.2)],
         'c': [('c', 6.9)]}
DICT3 = {'a': [('a3', 0.8), ('a1', 0.0)],
         'b': [('b1', 0.5), ('b2', 0.2)],
         'c': [('c', 6.9)]}
DICT4 = {'a': [('a1', 0.05), ('a2', 0.88)],
         'b': [('b1', 0.5), ('b2', 0.2), ('b3', 0)],
         'c': [('c', 6.95)]}
DICT5 = {'a': [('a1', 0.1), ('a2', 0.9)],
         'b': [('b1', 0.5), ('b2', 0.2)],
         'c': [('c', 6.95)],
         'd': [('d1', 6)]}
DICT6 = {'a': [('a1', 0.1), ('a2', 0.9)],
         'b': [('b1', 0.5), ('b2', 0.2)],
         'c': [('c', 6.89)]}

DICT_I1 = {'a': [('a1', np.array([0.123])), ('a2', (0.001, 0.002))],
           'b': [('b1', 0.505), ('b2', 0.201)],
           'c': [('c', 6.008)]}
DICT_I2 = {'a': [('a2', [0.000, 0.003]), ('a1', (0.123, ))],
           'b': [('b1', 0.504), ('b2', 0.2)],
           'c': [('c', 6.009)]}
# yapf: enable

DICT1_ = {k: dict(v) for k, v in DICT1.items()}
DICT2_ = {k: dict(v) for k, v in DICT2.items()}
DICT3_ = {k: dict(v) for k, v in DICT3.items()}
DICT4_ = {k: dict(v) for k, v in DICT4.items()}
DICT5_ = {k: dict(v) for k, v in DICT5.items()}
DICT6_ = {k: dict(v) for k, v in DICT6.items()}
DICT_I1_ = {k: dict(v) for k, v in DICT_I1.items()}
DICT_I2_ = {k: dict(v) for k, v in DICT_I2.items()}


def test_invalid_models():
    """
    Tests the invalid models: ``InvalidModel`` and ``NonProbabilisticModel``.
    """
    invalid_model = futt.InvalidModel()
    assert isinstance(invalid_model, object)
    assert invalid_model.fit(None, None) is None

    string = 'prediction_function 1'

    def p_func(x):
        return string.format(x)

    non_probabilistic_model = futt.NonProbabilisticModel(p_func)
    assert non_probabilistic_model.prediction_function(1) == string
    assert non_probabilistic_model.predict(1) == string


def test_is_explanation_equal_list():
    """
    Tests :func:`fatf.utils.testing.transparency.is_explanation_equal_list`.
    """
    assert futt.is_explanation_equal_list(DICT1, DICT1)
    assert futt.is_explanation_equal_list(DICT1, DICT2)
    assert not futt.is_explanation_equal_list(DICT1, DICT3)
    assert not futt.is_explanation_equal_list(DICT1, DICT4)
    assert not futt.is_explanation_equal_list(DICT1, DICT5)
    assert not futt.is_explanation_equal_list(DICT1, DICT6)
    # Different tolerance
    assert futt.is_explanation_equal_list(DICT1, DICT1, atol=1e-5)
    assert not futt.is_explanation_equal_list(DICT1, DICT2, atol=1e-2)
    # Iterable content
    assert futt.is_explanation_equal_list(DICT_I1, DICT_I1, atol=1e-3)
    assert futt.is_explanation_equal_list(DICT_I1, DICT_I2, atol=1e-3)
    assert not futt.is_explanation_equal_list(DICT_I1, DICT_I2, atol=1e-4)

    assert futt.is_explanation_equal_list(DICT2, DICT1)
    assert futt.is_explanation_equal_list(DICT2, DICT2)
    assert not futt.is_explanation_equal_list(DICT2, DICT3)
    assert not futt.is_explanation_equal_list(DICT2, DICT4)
    assert not futt.is_explanation_equal_list(DICT2, DICT5)
    assert futt.is_explanation_equal_list(DICT2, DICT6)

    assert not futt.is_explanation_equal_list(DICT3, DICT1)
    assert not futt.is_explanation_equal_list(DICT3, DICT2)
    assert futt.is_explanation_equal_list(DICT3, DICT3)
    assert not futt.is_explanation_equal_list(DICT3, DICT4)
    assert not futt.is_explanation_equal_list(DICT3, DICT5)
    assert not futt.is_explanation_equal_list(DICT3, DICT6)

    assert not futt.is_explanation_equal_list(DICT4, DICT1)
    assert not futt.is_explanation_equal_list(DICT4, DICT2)
    assert not futt.is_explanation_equal_list(DICT4, DICT3)
    assert futt.is_explanation_equal_list(DICT4, DICT4)
    assert not futt.is_explanation_equal_list(DICT4, DICT5)
    assert not futt.is_explanation_equal_list(DICT4, DICT6)

    assert not futt.is_explanation_equal_list(DICT5, DICT1)
    assert not futt.is_explanation_equal_list(DICT5, DICT2)
    assert not futt.is_explanation_equal_list(DICT5, DICT3)
    assert not futt.is_explanation_equal_list(DICT5, DICT4)
    assert futt.is_explanation_equal_list(DICT5, DICT5)
    assert not futt.is_explanation_equal_list(DICT5, DICT6)

    assert not futt.is_explanation_equal_list(DICT6, DICT1)
    assert futt.is_explanation_equal_list(DICT6, DICT2)
    assert not futt.is_explanation_equal_list(DICT6, DICT3)
    assert not futt.is_explanation_equal_list(DICT6, DICT4)
    assert not futt.is_explanation_equal_list(DICT6, DICT5)
    assert futt.is_explanation_equal_list(DICT6, DICT6)


def test_is_explanation_equal_dict():
    """
    Tests :func:`fatf.utils.testing.transparency.is_explanation_equal_dict`.
    """
    assert futt.is_explanation_equal_dict(DICT1_, DICT1_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT1_, DICT2_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT1_, DICT3_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT1_, DICT4_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT1_, DICT5_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT1_, DICT6_, atol=1e-1)
    # Different tolerance
    assert futt.is_explanation_equal_dict(DICT1_, DICT1_, atol=1e-5)
    assert not futt.is_explanation_equal_dict(DICT1_, DICT2_, atol=1e-2)
    # Iterable content
    assert futt.is_explanation_equal_dict(DICT_I1_, DICT_I1_)
    assert futt.is_explanation_equal_dict(DICT_I1_, DICT_I1_, atol=1e-3)
    assert futt.is_explanation_equal_dict(DICT_I1_, DICT_I2_)
    assert not futt.is_explanation_equal_dict(DICT_I1_, DICT_I2_, atol=1e-4)

    assert futt.is_explanation_equal_dict(DICT2_, DICT1_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT2_, DICT2_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT2_, DICT3_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT2_, DICT4_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT2_, DICT5_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT2_, DICT6_, atol=1e-1)

    assert not futt.is_explanation_equal_dict(DICT3_, DICT1_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT3_, DICT2_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT3_, DICT3_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT3_, DICT4_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT3_, DICT5_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT3_, DICT6_, atol=1e-1)

    assert not futt.is_explanation_equal_dict(DICT4_, DICT1_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT4_, DICT2_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT4_, DICT3_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT4_, DICT4_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT4_, DICT5_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT4_, DICT6_, atol=1e-1)

    assert not futt.is_explanation_equal_dict(DICT5_, DICT1_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT5_, DICT2_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT5_, DICT3_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT5_, DICT4_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT5_, DICT5_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT5_, DICT6_, atol=1e-1)

    assert not futt.is_explanation_equal_dict(DICT6_, DICT1_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT6_, DICT2_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT6_, DICT3_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT6_, DICT4_, atol=1e-1)
    assert not futt.is_explanation_equal_dict(DICT6_, DICT5_, atol=1e-1)
    assert futt.is_explanation_equal_dict(DICT6_, DICT6_, atol=1e-1)
