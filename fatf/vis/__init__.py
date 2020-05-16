"""
The :mod:`fatf.vis` module implements visualisations of various FAT methods.

**This module requires the matplotlib package to be installed.**
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        'matplotlib Python module is not installed on your system. '
        'You must install it in order to use fatf.vis functionality. '
        'One possibility is to install matplotlib alongside this package via '
        'visualisation dependencies with: pip install fat-forensics[vis].')
else:
    # Setup matplotlib style
    plt.style.use('seaborn')
    # Cleanup
    del plt
    del matplotlib
