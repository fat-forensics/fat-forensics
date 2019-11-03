"""
.. deprecated:: 0.0.3
   This module will be deprecated in FAT Forensics version 0.0.3.
   Instead of wrapping the lime package a full (modular) version of the
   LIME surrogate explainer has been implemented -- see the :class:`fatf.\
transparency.predictions.surrogate_explainers.TabularBlimeyLime` class and the
   :ref:`how_to_tabular_surrogates` how-to guide for more details.

The :mod:`fatf.transparency.predictions.lime` module prepares the LIME
explainer for explaining predictions.

This module instantiates :class:`fatf.transparency.lime.Lime` class with
selected arguments to create a :class:`.Lime` class explainer for a single
prediction.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

import numpy as np

import fatf.transparency.lime as ftl

__all__ = ['Lime']


class Lime(ftl.Lime):
    """
    A sub-class of the Lime class designated for explaining a prediction.

    This class initialises a :class:`fatf.transparency.lime.Lime` class and
    overwrites the ``sample_around_instance``/``local_explanation`` to be
    ``True``, therefore creates a single prediction explainer.

    For reference please see the description of the
    :class:`fatf.transparency.lime.Lime` class.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, data: np.ndarray, model: object = None,
                 **kwargs: bool) -> None:
        """
        Initialises a tabular LIME wrapper for a prediction explanation.
        """
        sampling_param = 'sample_around_instance'
        wmsg = ('Disregarding the {} parameter -- this LIME tabular explainer '
                'object should only be used to explain a prediction. If you '
                'are interested in explaining a model, please refer to the '
                'fatf.transparency.models.lime module.'.format(sampling_param))

        params = dict(kwargs)
        if sampling_param in params:
            if bool(params[sampling_param]) is False:
                warnings.warn(wmsg, UserWarning)
            del params[sampling_param]
            assert sampling_param not in params

        super().__init__(data, local_explanation=True, model=model, **params)
