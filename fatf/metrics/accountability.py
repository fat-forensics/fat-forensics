"""Privacy and security metrics.

This module gathers various metrics to assess privacy and security of
an artificial intelligence pipeline.

TODO: Implement:
    * k-anonymity,
    * l-diversity, and
    * t-closeness.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class kanonymity():
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        k (int): k parameter in k-anonymity. Defaults to 5.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """

    def __init__(self, k=5):
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        return 0

    def example_function(self, x):
        """Example function docstring.

        Args:
            x (np.ndarray): 

        Returns:
            int: Returns 0 if successful.

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions
                that are relevant to the interface.

        Examples:
            Examples should be written in doctest format, and should illustrate how
            to use the function.

            >>> print(kanonymity.example_function(None))
            0
        """
        return 0
