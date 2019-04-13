"""
This package is responsible for a visualisation of various FAT methods.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

try:
    import matplotlib
except ImportError:
    _warning_msg = (  # pylint: disable=invalid-name
        'matplotlib Python module is not installed on your system. '
        'You must install it in order to use fatf.vis functionality. '
        'One possibility is to install matplotlib alongside this package via '
        'visualisation dependencies with: pip install fatf[vis].')
    warnings.warn(_warning_msg, ImportWarning)
else:
    # Setup matplotlib style
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    del plt

    del matplotlib
finally:
    del warnings
