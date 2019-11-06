.. title:: Tutorials

.. _tutorials:

Tutorials
+++++++++

The FAT Forensics tutorials aim to get you up to speed and build up your
confidence in using the package. They are intended for the beginners and each
tutorials aims to teach you how to solve a particular FAT-related issue of a
predictive modeling task: data, models and/or predictions.

Before continuing, please make sure that you have
:ref:`installed <installation_instructions>` the FAT Forensics package and you
are ready to follow the tutorials. To check whether :mod:`fatf` is installed,
launch a Python interpreter and check the version of the package::

  $ python
  >>> import fatf
  >>> fatf.__version__
  '0.0.2'

.. note:: **Doctest Mode**

   The code-examples in the :ref:`tutorials` and :ref:`how_to_guide` are
   written in a *python-console* format. If you wish to easily execute these
   examples in **IPython**, use the::

   %doctest_mode

   `magic command`_ in the IPython-console. You can then simply copy and paste
   the examples directly into IPython without having to worry about removing
   the `>>>` manually.

.. _`magic command`: https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-doctest_mode

Tutorials Content
=================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :name: tutorialstoc

   grouping
   grouping-fairness
   grouping-robustness
   model-explainability
   prediction-explainability
