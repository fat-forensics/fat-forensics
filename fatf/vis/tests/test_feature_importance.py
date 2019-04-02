"""
Tests the exceptions raised by feature importance functions
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest
import numpy as np 

from fatf.exceptions import IncorrectShapeError
from fatf.vis import feature_importance as visfeat


ice_matrix = np.ones((3, 10, 3))
pd_matrix = np.ones((10, 3))
values = np.ones((10,))
err_values = np.ones((20,))

ice_shape_err = ('plot_individual_condtional_expectation expects matrix of '
                 'shape [n_samples, steps, n_classes].')
pd_shape_err = ('plot_partial_depenedence expects matrix of shape [steps, '
                'n_classes].')
category_err = ('Category {} is not a valid index for probability '
                'matrix.'.format(10))
values_err = ('{} values provided does not match {} steps in '
              'probability matrix.'.format(20, 10))


def test_plot_individual_conditional_expectation():
    with pytest.raises(IncorrectShapeError) as exin:
        visfeat.plot_individual_conditional_expectiation(pd_matrix, values, 0)
    assert str(exin.value) == ice_shape_err

    with pytest.raises(ValueError) as exin:
        visfeat.plot_individual_conditional_expectiation(ice_matrix, values, 10)
    assert str(exin.value) == category_err

    with pytest.raises(ValueError) as exin:
        visfeat.plot_individual_conditional_expectiation(
            ice_matrix, err_values, 0)
    assert str(exin.value) == values_err


def test_plot_partial_dependence():
    with pytest.raises(IncorrectShapeError) as exin:
        visfeat.plot_partial_dependence(ice_matrix, values, 0)
    assert str(exin.value) == pd_shape_err

    with pytest.raises(ValueError) as exin:
        visfeat.plot_partial_dependence(pd_matrix, values, 10)
    assert str(exin.value) == category_err

    with pytest.raises(ValueError) as exin:
        visfeat.plot_partial_dependence(pd_matrix, err_values, 0)
    assert str(exin.value) == values_err
