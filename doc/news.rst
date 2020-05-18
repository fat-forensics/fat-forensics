.. title:: News

.. _news:

News
++++

[18/05/2020] Version 0.1.0 released!
====================================

Today we release an incremental update focused on surrogate explainers.
This release coincides with publication of a `paper <JOSS_paper_>`_
describing FAT Forensic in The Journal of Open Source Software (JOSS).
The changelog summarises the functionality added with this release:
:ref:`changelog_0_1_0`.

Next up, surrogate explainability for image and text data.

.. _JOSS_paper: https://joss.theoj.org/papers/070c8b6b705bb47d1432673a1eb03f0c

[04/11/2019] Version 0.0.2 released!
====================================

Today we release an incremental update focused on surrogate explainability of
black-box models for tabular data -- a collection of techniques and algorithms
that we call *build LIME yourself* (**bLIMEy**). The changelog summarises the
functionality added with this release: :ref:`changelog_0_0_2`.

This code release comes with a new *how-to* guide:
:ref:`how_to_tabular_surrogates`. We have also added one more code example --
:ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_tree.py` --
and updated the LIME code example
(:ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_lime.py`)
to use the bLIMEy implementation of LIME
(:class:`fatf.transparency.predictions.surrogate_explainers.TabularBlimeyLime`)
instead. The ":ref:`tutorials_prediction_explainability`" tutorial has also
been updated to use bLIMEy instead of LIME.

Surrogate explainability for image and text data is coming soon so stay tuned.

[01/08/2019] Version 0.0.1 released!
====================================

Today we release our first public version of the FAT Forensics package. To
familiarise yourself with the FAT Forensics Python package make sure to check
out our :ref:`getting_started` page. The changelog summarises functionality
of this release: :ref:`changelog_0_0_1`.
