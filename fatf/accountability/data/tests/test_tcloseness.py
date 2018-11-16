# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:23:40 2018

@author: Rafael Poyiadzi
"""

import pytest
import numpy as np

    
from funcs import generalize_dates

# =============================================================================
# @pytest.mark.parametrize("dataset", [testdata])
# def test_tcloseness(dataset):
#     t = 2
#     divergence = KL
#     _, counts = np.unique(dataset['diagnosis'], return_counts = True)
#     probs = counts / sum(counts)
#     qi_combinations = list(itertools.product(*quasi_identifiers.values()))
#     for qi_combination in qi_combinations:
#         newD = filter_table(dataset, qi_combination)
#         div = divergence(newD['diagnosis'], probs)
#         assert div <= t
# =============================================================================

input1 = ['12/11/2005', '12/11/2005', '12/11/2005']
expected1 = ['12/11/2005', '12/11/2005', '12/11/2005']

input2 = ['13/11/2005', '12/11/2005', '11/11/2005']
expected2 = ['11-20/11/2005', '11-20/11/2005', '11-20/11/2005']

input3 = ['13/10/2005', '12/11/2005', '11/12/2005']
expected3 = ['*/10-12/2005', '*/10-12/2005', '*/10-12/2005']

input4 = ['13/6/2005', '12/11/2005', '11/2/2005']
expected4 = ['*/*/2005', '*/*/2005', '*/*/2005']

input5 = ['13/6/2005', '12/11/2006', '11/2/2007']
expected5 = ['*/*/200*', '*/*/200*', '*/*/200*']

@pytest.mark.parametrize("dates_list, output_dates", [(input5, expected5),
                                                      (input1, expected1),
                                                      (input2, expected2),
                                                      (input3, expected3),
                                                      (input4, expected4)])
def test_generalize_dates(dates_list, output_dates):
    out = generalize_dates(dates_list)
    print(out, 'hi')
    assert output_dates == out