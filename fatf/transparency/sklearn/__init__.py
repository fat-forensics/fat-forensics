"""
The :mod:`fatf.transparency.sklearn` module implements scikit-learn explainers.

**This module requires the scikit-learn package to be installed.**
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

try:
    import sklearn
except ImportError:
    _warning_msg = (  # pylint: disable=invalid-name
        'scikit-learn (sklearn) Python module is not installed on your '
        'system. You must install it in order to use '
        'fatf.transparency.sklearn functionality. '
        'One possibility is to install scikit-learn alongside this package '
        'via machine learning dependencies with: pip install fatf[ml].')
    warnings.warn(_warning_msg, ImportWarning)
else:
    del sklearn
finally:
    del warnings
