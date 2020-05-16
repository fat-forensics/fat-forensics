"""
The :mod:`fatf.transparency.sklearn` module implements scikit-learn explainers.

**This module requires the scikit-learn package to be installed.**
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

try:
    import sklearn
except ImportError:
    raise ImportError(
        'scikit-learn (sklearn) Python module is not installed on your '
        'system. You must install it in order to use '
        'fatf.transparency.sklearn functionality. '
        'One possibility is to install scikit-learn alongside this package '
        'via machine learning dependencies with: pip install '
        'fat-forensics[ml].')
else:
    del sklearn
