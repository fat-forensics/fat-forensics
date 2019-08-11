.. title:: Defining Sub-Groups in Your Data

.. _tutorials_grouping:

Exploring the Grouping Concept -- Defining Sub-Populations
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. topic:: Tutorial Contents

    In this tutorial, we introduce the concept of sub-populations that is very
    useful in a number of :ref:`Fairness <fairness_examples>` and
    :ref:`Accountability <accountability_examples>`. The idea is to find
    indices of data points (rows) or the data points themselves that form
    separate subpopulations, which may be underrepresented in the training data
    or disparately treated by a predictive model.

As always, we start by importing FAT Forensics inside a Python interpreter::

   $ python
   >>> import fatf

We also import numpy and pretty print (``pprint``) as we will need these for
the tutorial::

   >>> import numpy as np
   >>> from pprint import pprint

There are two different types of features that our data can be grouped on:

* categorical -- usually string-based or numerical with a finite number of
  possible values with groups being defined by *disjoint* sub-sets of that
  feature space, or
* numerical -- integer- or floating point-based with the groups defined by
  ranges on the feature space.

Data Grouping
=============

Grouping Based on a Numerical Feature
-------------------------------------

The very basic grouping that one may desire is based on the unique values of
the *target vector* for a classification task. Such grouping can be useful when
inspecting the distribution of certain features between different classes.

Before we continue, let us load the iris data set, which we will use for this
tutorial::

   >>> import fatf.utils.data.datasets as fatf_datasets
   >>> iris_data_dict = fatf_datasets.load_iris()
   >>> iris_data = iris_data_dict['data']

The *target array* can be accessed via the ``'target'`` key in the
``iris_data_dict`` dictionary::

   >>> iris_target = iris_data_dict['target'].astype(int)

   >>> iris_target.shape
   (150,)

Since the ``iris_target`` vector is 1-dimensional and the grouping function
(:func:`fatf.utils.data.tools.group_by_column`) requires a data set, i.e. a
2-dimensional numpy array, we need to reshape it to ``(150, 1)`` first::

   >>> iris_target_2d = iris_target.reshape((150, 1))
   >>> iris_target_2d.shape
   (150, 1)

Now, that we have the target array in the right shape, let us see what are all
the unique values that it holds::

   >>> np.unique(iris_target_2d)
   array([0, 1, 2])

Since the target array is a numerical array, when we use the
:func:`fatf.utils.data.tools.group_by_column` function, it will infer that the
type of the column is *numerical* and treat it as such, which will lead to
binning the target vector rather than grouping it based on its unique values
(the second parameter in the function call below indicates that we want to
perform the grouping based on the 1st column, which index is ``0``)::

   >>> import fatf.utils.data.tools as fatf_data_tools
   >>> target_grouping_num = fatf_data_tools.group_by_column(iris_target_2d, 0)

.. module::
   fatf.utils.data.tools
   :noindex:

The :func:`group_by_column` function returns a 2-tuple with the first element
being the grouping (row indices) and the second being group names. Let us see
what groups we got::

   >>> target_grouping_num[1]
   ['x <= 0.4', '0.4 < x <= 0.8', '0.8 < x <= 1.2000000000000002', '1.2000000000000002 < x <= 1.6', '1.6 < x']

As expected, the grouping is numerical with 5 bind (the default). To get the
desired binning -- three groups, one with 0s, one with 1s and the last one with
2s -- we can either modify the bin boundaries or force the
:func:`group_by_column` function to treat this numerical vector as a
categorical one. Let us start with the custom numerical binning::

   >>> target_grouping_num_custom = fatf_data_tools.group_by_column(
   ...     iris_target_2d, 0, groupings=[0.5, 1.5])
   >>> target_grouping_num_custom[1]
   ['x <= 0.5', '0.5 < x <= 1.5', '1.5 < x']

This binning should create 3 bins, each with 50 indices; the first one with
indices between 0 and 49, the second one with indices between 50 and 99 and the
last one with indices between 100 and 149 as this is how the labels are
distributed (they are ordered, to see this you can inspect the ``iris_target``
variable)::

   >>> len(target_grouping_num_custom[0])
   3
   >>> target_grouping_num_custom[0][0]
   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
   >>> target_grouping_num_custom[0][1]
   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
   >>> target_grouping_num_custom[0][2]
   [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

Grouping Based on a Numerical Feature -- Treat as a Categorical
---------------------------------------------------------------

Instead of this elaborate workaround we can simply setting the
``treat_as_categorical`` parameter to ``True`` and the numerical vector will
be treated as a categorical one, therefore grouping based on the unique
values in that vector::

   >>> target_grouping_num_cat = fatf_data_tools.group_by_column(
   ...     iris_target_2d, 0, treat_as_categorical=True)
   >>> target_grouping_num_cat[1]
   ['(0,)', '(1,)', '(2,)']

Therefore, the first group has indices of the target value equal to 1, etc.
Now, let us see whether the grouping is the same as the one that we got in
``target_grouping_num_custom``::

   >>> np.array_equal(target_grouping_num_custom[0][0],
   ...                np.sort(target_grouping_num_cat[0][0]))
   True
   >>> np.array_equal(target_grouping_num_custom[0][1],
   ...                np.sort(target_grouping_num_cat[0][1]))
   True
   >>> np.array_equal(target_grouping_num_custom[0][2],
   ...                np.sort(target_grouping_num_cat[0][2]))
   True

You may wonder why the groups are defined as tuples with just one element, i.e.
``'(0,)'``. This is because the categorical groupings can be customised to
include multiple values in a single group. To see how that works, let us group
``0`` and ``1`` together with the other group including just ``2``. To this
end, we use the ``groupings`` parameter again but this time in a manner
that is recognised by a categorical grouping::

   >>> target_grouping_num_cat_custom = fatf_data_tools.group_by_column(
   ...     iris_target_2d,
   ...     0,
   ...     treat_as_categorical=True,
   ...     groupings=[(0, 1), (2, )])
   >>> target_grouping_num_cat_custom[1]
   ['(0, 1)', '(2,)']

That looks promising, let us see whether the grouping is correct::

   >>> np.array_equal(np.sort(target_grouping_num_cat_custom[0][0]),
   ...                list(range(100)))
   True
   >>> np.array_equal(np.sort(target_grouping_num_cat_custom[0][1]),
   ...                list(range(100, 150)))
   True

.. _tutorials_grouping_data_transparency:

Using Grouping to Inspect a Data Set -- Data Transparency
=========================================================

Now that we know how to create groupings of data sets we can use this knowledge
to inspect transparency of the Iris data set. First, let us choose one of the
features -- ``'petal length (cm)'`` with index ``2`` -- and see how it is
distributed for the whole data set::

   >>> arbitrary_feature_index = 2
   >>> arbitrary_feature_name = \
   ...     iris_data_dict['feature_names'][arbitrary_feature_index]
   >>> arbitrary_feature_name
   'petal length (cm)'

Let us see how this feature is distributed for the whole data set using the
:func:`fatf.transparency.data.describe_functions.describe_array` function::

   >>> import fatf.transparency.data.describe_functions as fatf_describe_data
   >>> petal_length_desc = fatf_describe_data.describe_array(iris_data[:, 2])
   >>> pprint(petal_length_desc)
   {'25%': 1.600000023841858,
    '50%': 4.3500001430511475,
    '75%': 5.099999904632568,
    'count': 150,
    'max': 6.9,
    'mean': 3.7580001,
    'min': 1.0,
    'nan_count': 0,
    'std': 1.7594041}

Therefore, we have 150 data points with values ranging between ``1.0`` and
``6.9``. The mean of this feature for the whole data set is ``3.76`` while the
median is ``4.35``. Let us compare it against the distribution of this feature
for each of the three classes. To this end, we first need to split the Iris
data set using the grouping that we created in the previous section::

   >>> petal_length_class_0 = iris_data[target_grouping_num_cat[0][0],
   ...                                  arbitrary_feature_index]
   >>> petal_length_class_1 = iris_data[target_grouping_num_cat[0][1],
   ...                                  arbitrary_feature_index]
   >>> petal_length_class_2 = iris_data[target_grouping_num_cat[0][2],
   ...                                  arbitrary_feature_index]

Now, we can create a description of the ``'petal length (cm)'`` feature for
each class separately::

   >>> petal_length_class_0_desc = fatf_describe_data.describe_array(
   ...     petal_length_class_0)
   >>> print(iris_data_dict['target_names'][0])
   setosa
   >>> pprint(petal_length_class_0_desc)
   {'25%': 1.399999976158142,
    '50%': 1.5,
    '75%': 1.5750000178813934,
    'count': 50,
    'max': 1.9,
    'mean': 1.462,
    'min': 1.0,
    'nan_count': 0,
    'std': 0.1719186}

   >>> petal_length_class_1_desc = fatf_describe_data.describe_array(
   ...     petal_length_class_1)
   >>> print(iris_data_dict['target_names'][1])
   versicolor
   >>> pprint(petal_length_class_1_desc)
   {'25%': 4.0,
    '50%': 4.3500001430511475,
    '75%': 4.599999904632568,
    'count': 50,
    'max': 5.1,
    'mean': 4.26,
    'min': 3.0,
    'nan_count': 0,
    'std': 0.46518815}

   >>> petal_length_class_2_desc = fatf_describe_data.describe_array(
   ...     petal_length_class_2)
   >>> print(iris_data_dict['target_names'][2])
   virginica
   >>> pprint(petal_length_class_2_desc)
   {'25%': 5.099999904632568,
    '50%': 5.549999952316284,
    '75%': 5.8750001192092896,
    'count': 50,
    'max': 6.9,
    'mean': 5.552,
    'min': 4.5,
    'nan_count': 0,
    'std': 0.54634786}

The results displayed above show that the ``'petal length (cm)'`` feature is a
good predictor of the iris plant type. The *setosa* type has petals of length
between 1 and 1.9 centimeters, whereas the petals of the *versicolor* type are
between 3 and 5.1 centimeters and the *virginica* type has petals in the range
of 4.5 and 6.9 centimeters. This tells us that petal length can help us to
distinguish between the *setosa* type and the other two types, however by just
using this feature we cannot tell apart the *versicolor* and *virginica* types.

----

Getting all these insights was possible because of a data grouping based on the
target value (iris type). In addition to **data transparency** we can use
groupings to inspect *fairness* and *accountability* of data sets and
predictive models -- group-based disparity metrics and data sampling issues/\
robustness of a predictive model respectively.

The next two tutorials show how to use grouping for *fairness*
(:ref:`tutorials_grouping_fairness`) and *accountability*
(:ref:`tutorials_grouping_robustness`) of data and models. To learn more about
transparency you can have a look at the :ref:`tutorials_model_explainability`
or the :ref:`tutorials_prediction_explainability` tutorial or code examples
referenced therein.

Relevant FAT Forensics Examples
===============================

The following examples provide more structured and code-focused use-cases of
the data grouping function and the data description functionality:

* :ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_data_desc.py`,
* :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_models_measure.py`.
