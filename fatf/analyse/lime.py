"""
A wrapper for the LIME package to work with tabular data 
Author: Alex Hepburn <ah13558@bristol.ac.uk>
License: new BSD
"""

import sys
import numpy as np 
# do i need to import Dataset class or not
try:
    import lime
    import lime.lime_tabular
except ImportError as e:
    raise ImportError('Lime class requires LIME package to be installed. This can be installed by: '
        'pip install lime')
# install all wrapper depencies - fatf.install.something
# with default options 


class Lime():
    '''
    Wrapper for package implemented in https://github.com/marcotcr/lime
    # implemented as a class so you can use the same lime object with multiple instances
    Attributes:
        dataset: Dataset object that contains the dataset to test
        predictor: object that contains method predict(x) that outputs predictions
            and predict_proba(x) that outputs probability vectors corresponding to
            the probability of an instance belonging to each class
        categorical: list of strings or ints that are feature names in dataset that user 
            would like to specify as categorical
        num_samples: int number of samples to generate around x in LIME algorithm
            default: 5000
        num_features: int number of features to use in LIME algorithm (takes top-n features)
            default: None to use all features
        distance_metric: string defining distance to use in LIME algorithm, has to be valid
            metric for use in scipy.spatial.distance.pdist. default: 'euclidean'

    Example:
        >>>from sklearn.linear_model import LogisticRegression
        >>>predictor = LogisticRegression()
        >>>data = Dataset.from_csv('data.csv')
        >>>predictor.fit(data.X, data.target)
        >>>l = Lime(data, predictor, categorical=['gender'])
        >>>figure = l.explain_instance([30, 10, 0, 50], mode='figure')
    
    Raises:
        ValueError: if categorical feature not in dataset.header
    '''

    def __init__(self, dataset, predictor, categorical=None, num_samples=5000, 
                 num_features=None, distance_metric='euclidean'):
        self._dataset = dataset
        self.predictor = predictor
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self._process_categorical(categorical)
        if not self.num_features:
            self.num_features = self.dataset.X.shape[1]
        self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.dataset.X, feature_names=self.dataset.header, 
            categorical_features=self._categorical, class_names=self.dataset.class_names, 
            discretize_continuous=True)

    def __str__(self):
        return str(self.as_dict)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        '''
        Setter for dataset so we can reinitialise self.tabular_explainer  
        '''
        self._dataset = dataset
        self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
            self._dataset.X, feature_names=self._dataset.header, 
            categorical_features=self._categorical, class_names=self._dataset.class_names, 
            discretize_continuous=True)

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, categorical):
        '''
        Setter for categorical variable so we can reinitialise self.tabular_explainer  
        '''
        self._process_categorical(categorical)
        self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.dataset.X, feature_names=self._dataset.header, 
                categorical_features=self._categorical, class_names=self.dataset.class_names, 
                discretize_continuous=True)

    def _process_categorical(self, categorical):
        '''
        Checking list categorical is valid and sets self._categorical variable
        
        Args:
            categorical: list of strings or ints correpsonding to names or indicies of 
                feature names in dataset.header
        '''
        if categorical:
            if type(categorical[0]) == int:
                for i in categorical:
                    if i>len(self._dataset.header)-1:
                        raise ValueError('%d not a valid feature index in dataset.'%i)
                self._categorical = categorical
            else:
                categorical_indices = []
                for s in categorical:
                    if s not in self._dataset.header:
                        raise ValueError('%s not a valid feature name in dataset.'%s)
                    categorical_indices.append(self._dataset.header.index(s))
                self._categorical = categorical_indices
        else:
            self._categorical = categorical

    def explain_instance(self, x, mode='figure', labels=None):
        '''
        Uses LIME tabular_explainer to explain instance x and outputs in mode

        Args:
            x: np.array of instance to explain
            mode: string defining output mode to use. 'figure' outputs pyplot figure object,
                'dict' outputs dictionary, 'html' outputs html, 'notebook' shows explainer
                in jupyter notebook 
            labels: tuple of int labels to explain decisions for. If None, then all labels
                are used. Only used in modes 'figure'
        
        Returns:
            explainer depending on specified mode. Can be pyplot figre, dict, html string

        Raises:
            ValueError: string for mode not a valid mode
        '''
        if not labels:
            labels = tuple(set(self.dataset.target))
        self._exp = self.tabular_explainer.explain_instance(
                x, self.predictor.predict_proba, labels=labels, 
                num_features=self.num_features, num_samples=self.num_samples, 
                distance_metric=self.distance_metric)
        if mode == 'figure':
            return self.show_pyplot(labels=labels)
        elif mode == 'dict':
            return self.as_dict()
        elif mode == 'html':
            return self.as_html()
        elif mode == 'notebook':
            return self.show_notebook()
        else:
            raise ValueError('%s not a valid output mode.'%mode)
    
    def show_notebook(self):
        self._exp.show_in_notebook(predict_proba=True)

    def show_pyplot(self, labels=None):
        '''
        Figures to display explainer

        Args:
            labels: tuple of labels to explain decisions for. If None, then all labels
                are used.

        Returns: Figure from matplotlib where it is split into as many subplots as there are 
            possible labels in Dataset.
        '''
        import matplotlib.pyplot as plt
        sharey = False
        names = None
        if not labels:
            labels = sorted(set(self.dataset.target)) # use all labels
        if self.num_features == self.dataset.X.shape[1]:
            # If all features are used then every ytick will be the same for each subplot
            sharey = True 
            f, axs = plt.subplots(1, len(set(self.dataset.target)), sharey=sharey)
        else:
            f, axs = plt.subplots(len(set(self.dataset.target)), 1, sharex=True)
        f.suptitle('Local Explanations for classes')
        if sharey: # Make sure all barplots are in the same order if sharing
            exp = self.as_list(label=0)
            names = [x[0] for x in exp]
        for ax, label in zip(axs, labels):
            ax = self._as_pyplot_figure(label, ax, names)
        return f

    def _as_pyplot_figure(self, label, axes, names=None):
        '''TAKEN FROM https://github.com/marcotcr/lime/blob/master/lime/explanation.py
        Edited to take axes for plotting subplots
        Returns the explanation as a pyplot figure.
        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            axes: subplot axis to plot the bar chart on.
            names: names of features to use as yticks. If None, then the yaxis will not
                be shared between subplots and each tick will be given by the explainer.
        '''
        exp = self.as_list(label=label)
        vals = [x[1] for x in exp]
        unordered_names = [x[0] for x in exp]
        if not names:
            names = unordered_names
        else:
            # We need to order vals to be in the same order as corresponding names if 
            # y-axis is shared
            ind = [unordered_names.index(item) for item in names]
            vals = [vals[i] for i in ind]
        vals.reverse()
        labels = names[::-1]
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        axes.barh(pos, vals, align='center', color=colors)
        axes.set_yticks(pos)
        axes.set_yticklabels(labels)
        title = str(self._exp.class_names[label])
        axes.set_title(title)

    def as_dict(self):
        '''
        Returns dict where key is class_name and value is as_list
        '''
        d = {}
        class_names = self._exp.class_names
        for i in range(0, len(class_names)):
            d[class_names[i]] = self.as_list(label=i)
        return d

    def as_list(self, label=1):
        return self._exp.as_list(label)
    
    def as_html(self):
        return self._exp.as_html()
