"""
Wraps the fatf.transparency.lime.Lime explainer for a whole model.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

import numpy as np

import fatf.transparency.lime as ftl

__all__ = ['Lime']


class Lime(ftl.Lime):
    """
    A sub-class of the Lime class designated for explaining a whole model.

    This class initialises a :class:`fatf.transparency.lime.Lime` class and
    overwrites the ``sample_around_instance``/``local_explanation`` to be
    ``False``, therefore creates a surrogate explainer of the whole model.

    For reference please see the description of the
    :class:`fatf.transparency.lime.Lime` class.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, data: np.ndarray, model: object = None,
                 **kwargs: bool) -> None:
        """
        Initialises a tabular LIME wrapper for a model explanation.
        """
        sampling_param = 'sample_around_instance'
        wmsg = ('Disregarding the {} parameter -- this LIME tabular explainer '
                'object should only be used to explain a model. If you are '
                'interested in explaining a prediction, please refer to the '
                'fatf.transparency.predictions.lime '
                'module.'.format(sampling_param))

        params = dict(kwargs)
        if sampling_param in params:
            if bool(params[sampling_param]) is True:
                warnings.warn(wmsg, UserWarning)
            del params[sampling_param]
            assert sampling_param not in params

        super().__init__(data, local_explanation=False, model=model, **params)
