"""
The :mod:`fatf.accountability.data` module holds the objects and relevant
functions for anonymising datasets.
"""

# Author: Rafael Poyiadzis <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from numpy.lib import recfunctions as rfn
import math
from fatf.utils.validation import check_array_type

def lca_realline(X, rounding=5):
    X = list(map(int, X))
    max_val = max(X)
    min_val = min(X)
    lb = math.floor(int(min_val/rounding)*rounding)
    ub = math.ceil(int(max_val/rounding)*rounding)
    lca = str(lb) + '-' + str(ub)
    
    return lca 

def range_func_realline(X):
    X = list(map(int, X))
    max_val = max(X)
    min_val = min(X)
    return max_val - min_val

def get_distr(inputlist):
    unique, counts = np.unique(inputlist, return_counts=True)
    count = dict(zip(unique, counts))
    s = sum(count.values())
    for key in count.keys():
        count[key] /= s

    return count

def get_emd_fordistrs(distr1, distr2):
    union_set = set(distr1.keys()).union(distr2.keys())
    emd = 0
    for item in union_set:
        try:
            val1 = distr1[item]
            try:
                val2 = distr2[item]
                emd += np.abs(val1 - val2)
            except:
                emd += val1 
        except:
            emd += distr2[item]
    return emd
            
def get_emd_forlists(list1, list2):
    count1 = get_distr(list1)
    count2 = get_distr(list2)
    
    return get_emd_fordistrs(count1, count2)




class BaseAnonymiser(object):
    def __init__(self, 
                 dataset, 
                 identifiers,
                 quasi_identifiers,
                 sensitive_attributes,
                 lca_funcs = {},
                 range_funcs = {}):
        
        numerical_features, categorical_features = check_array_type(dataset)
        self.numerucal_features = numerical_features.tolist()
        self.categorical_features = categorical_features.tolist()
        self.features = self.numerucal_features + self.categorical_features
        
        self._dataset = dataset.copy(order='K')
        self._quasi_identifiers = quasi_identifiers
        self._attributes_to_suppress = identifiers
        self._sensitive_attributes = sensitive_attributes
        self.lca_funcs = lca_funcs
        self.range_funcs = range_funcs
        self.check_funcs()
        self.n_samples = self.dataset.shape[0]
        self.check_input()
        
        #TODO: potential issue with concat
        if len(self.sensitive_attributes) > 1:
            self.concatenate_sensitive_attributes()

        self.SA = self.dataset[self.sensitive_attributes[0]] 

    @property
    def cluster_assigments(self):
        return self.cluster_assignments
    
    @property
    def dataset(self):
        return self._dataset
    
    @property
    def quasi_identifiers(self):
        return self._quasi_identifiers
    
    @quasi_identifiers.setter
    def quasi_identifiers(self, quasi_identifiers):
        self._quasi_identifiers = quasi_identifiers
        self.check_input()
        
    @property
    def attributes_to_suppress(self):
        return self._attributes_to_suppress
    
    @attributes_to_suppress.setter
    def attributes_to_suppress(self, attributes_to_suppress):
        self._attributes_to_suppress = attributes_to_suppress
        self.check_input()
        
    @property
    def sensitive_attributes(self):
        return self._sensitive_attributes
    
    @sensitive_attributes.setter
    def sensitive_attributes(self, sensitive_attributes):
        self._sensitive_attributes = sensitive_attributes
        self.SA = self.dataset[self.sensitive_attributes[0]] 
        if len(self.sensitive_attributes) > 1:
            self.concatenate_sensitive_attributes()
    
    def check_funcs(self):
        if not self.lca_funcs:
            self.lca_funcs = dict(zip(self.features, [None]*len(self.features)))
        lca_keys = set(self.lca_funcs.keys())
        if not self.range_funcs:   
            self.range_funcs = dict(zip(self.features, [None]*len(self.features)))
            
        range_keys = set(self.range_funcs.keys())
        for field_name in self.features:
            if (field_name not in lca_keys or 
                    self.lca_funcs[field_name] is None):
                if field_name in self.numerucal_features:
                    self.lca_funcs[field_name] = lca_realline
                    lca_keys.add(field_name)
                    
            if (field_name not in range_keys or 
                    self.range_funcs[field_name] is None):
                if field_name in self.numerucal_features:
                    self.range_funcs[field_name] = range_func_realline
                    range_keys.add(field_name)
                    
            if field_name in self.quasi_identifiers:
                if (field_name not in lca_keys or 
                        self.lca_funcs[field_name] is None):
                    raise ValueError('missing lca function for %s: ', field_name)
                    
                if (field_name not in range_keys or 
                        self.range_funcs[field_name] is None):
                    raise ValueError('missing range function for %s: ', field_name)
    
    def change_numerical_to_str(self, data):
        for attr in self.numerucal_features:
            t = data[attr]
            data = rfn.drop_fields(data, attr)
            data = rfn.append_fields(data, attr, t, dtypes='<U9', usemask=False)
        return data
            
    def initialise_clustering(self):
        cluster_assignments = np.array(-np.ones(self.n_samples), dtype='int32')
        idx = np.random.randint(self.n_samples, size=1)    
        cluster_assignments[idx] = 0
        return cluster_assignments
        
    def get_total_information_loss(self, dataset):
        """ Calculates the total information loss for the current clustering.
        
        Description: Function that will calculate the total information loss,
                    for the current clustering, based on the 'features' provided.
        
        Args: 
            dataset: Structured Numpy Array dataset to be anonymized.
            features: List of features that will be used to calculate information loss.
            range_funcs: List of functions, one per feature, to be used to calculate
                        the range of each feature. For continuous, it is of the form
                        'MAX - MIN', while for categorical it is the height of the
                        tree.
            clustering_assignments: List of clustering assignments
            
        Returns: The total information loss for the current clustering.
        
        Raises:
            NA 
        """
        unique_clusters = list(set(self.cluster_assignments))
        full_dataset_ranges = self.get_feature_ranges(dataset)
    
        total_information_loss = 0
        for cluster in unique_clusters:
            if cluster == -1:
                continue
            filtered_dataset = self.filter_dataset(dataset, [cluster])
            total_information_loss += self.get_information_loss(filtered_dataset, full_dataset_ranges)
        return total_information_loss

    def get_information_loss(self, dataset, full_dataset_ranges):
        """ Calculates the information loss for the current cluster.
        
        Description: Function that will calculate the total information loss,
                    for the current clustering, based on the 'features' provided. 
                    For continuous variables, it is the range of the variable in the 
                    cluster (MAX - MIN) divided by the range in the full dataset, 
                    multiplied by the size of the cluster. For categorical it is 
                    the ratio of the heights of the trees, multiplied by the size
                    of the cluster.
        
        Args: 
            dataset: Structured Numpy Array dataset to be anonymized.
            features: List of features that will be used to calculate information loss.
            range_funcs: List of functions, one per feature, to be used to calculate
                        the range of each feature. For continuous, it is of the form
                        'MAX - MIN', while for categorical it is the height of the
                        tree.
            clustering_assignments: List of clustering assignments
            
        Returns: The information loss for the current cluster.
        
        Raises:
            NA 
        """
        dataset_length = dataset.shape[0]
        information_loss = 0
        for attr_name, attr_range_func in self.range_funcs.items():
            if attr_name in self.quasi_identifiers:
                attr_range = attr_range_func(dataset[attr_name])            
                information_loss += attr_range / full_dataset_ranges[attr_name]
        return int(dataset_length*information_loss)

    def get_feature_ranges(self, dataset):
        """
        Applied the range_funcs on each feature.
        """
        ranges = {}
        for attr_name, attr_range_func in self.range_funcs.items():
            if attr_name in self.quasi_identifiers:
                ranges[attr_name] = attr_range_func(dataset[attr_name])
        return ranges
     
    def filter_dataset(self, dataset, filters):
        """
        Returns the entries of the dataset for the specific clusters.
        """
        filter_list = []
        for value in self.cluster_assignments:
            if value in filters:
                filter_list.append(True)
            else:
                filter_list.append(False)
        return dataset[filter_list]
    
    def concatenate_sensitive_attributes(self, newfield_name = None, newfield_type = '<U16'):
        """ Combines sensitive attributes in the case of them being more than one.
        
        Description: Cross product between the sensitive attributes.
        
        Args: 
            dataset: Structured Numpy Array dataset to be anonymized.
            sensitive_attributes: List of strings representing the attributes to be combined.
            newfield_name: Option to provide the name of the new field. Default
                            is the concatenation of the two.
            newfield_type: Option to provide the dtype of the newfield. Default
                            is '<U30'.
                            
        Returns: The dataset provided but with the previous fields removed,
                and a new field - their combinarion - being included.
        
        Raises:
            NA
            """
        newcolumn = np.array(list(map(lambda x: x[0] + '-' + x[1], self.dataset[self.sensitive_attributes])))
        if not newfield_name:
            newfield_name = self.sensitive_attributes[0] + '-' + self.sensitive_attributes[1]
        else:
            if type(newfield_name) != str:
                newfield_name = str(newfield_name)
        self._dataset = rfn.append_fields(self.dataset, newfield_name, newcolumn, dtypes=newfield_type).data
        self._dataset = rfn.drop_fields(self.dataset, self.sensitive_attributes)
        self.sensitive_attributes = [newfield_name]
    
    def suppress(self):
        """ Suppression function
        
        Description: Will suppress all the values in 'dataset' under,
                    the fields provided in 'attributes_to_suppress'.
        
        Args: 
            dataset: Structured Numpy Array dataset to be anonymized.
            attributes_to_suppress: List of strings representing attributes,
                                    to be suppressed.
            
        Returns: The dataset provided but with the fields specified suppressed.
        
        Raises:
            ValueError when no attributes to be suppressed are provided.
            TypeError when the attributes to be provided are not strings.
            """
            
        if not self.attributes_to_suppress:
            raise ValueError('No attributes_to_suppress have been provided')
        
        for item in self.attributes_to_suppress:
            if type(item) is not str:
                raise TypeError('attributes_to_suppress should be of type "str" ')
                
        for attr in self.attributes_to_suppress:
            self.dataset[attr] = '*'
    
    def assign_extras(self):
        """ Function to alocate the remaining datapoints to existing clusters.
        
        Description: After the main clusters have been formed, this function will
                    alocate the remaining datapoints to them. For example, in 
                    k-anonymity this function will be called when the remaining instances
                    are fewer than k. For l-diversity it will be called when the 
                    remaining distinct elements for the Sensitive Attribute are 
                    fewer than l.
        
        Args: 
            dataset: Structured Numpy Array dataset to be anonymized.
            features: List of features that will be used to calculate information loss.
            range_funcs: List of functions, one per feature, to be used to calculate
                        the range of each feature. For continuous, it is of the form
                        'MAX - MIN', while for categorical it is the height of the
                        tree.
            clustering_assignments: List of clustering assignments
            
        Returns: Finalized cluster assignments.
        
        Raises:
            NA 
        """
        cluster_counter = int(max(self.cluster_assignments))
        while sum(self.cluster_assignments == -1) > 0:
            for idx, val in enumerate(self.cluster_assignments):
                if val != -1:
                    continue
                scores = []
                for cluster in range(cluster_counter+1):
                    self.cluster_assignments[idx] = cluster
                    scores.append((idx, 
                                   self.get_total_information_loss(self.dataset)))
                    self.cluster_assignments[idx] = -1
                best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
                self.cluster_assignments[best] = cluster
    
    def get_equivalence_classes(self):
        """ Function that will form the equivalence class (EQ) for each cluster.
        
        Description: For each of the clusters and for each of the features provided,
                    will compute the EQ using the lca_funcs provided by the user.
        
        Args: 
            data: Structured Numpy Array dataset to be anonymized.
            features: List of features that will be used to calculate information loss.
            lca_funcs: List of functions, one per feature, to be used to compute the
                        lowest common ancestor for each cluster for each feature.
            clustering_assignments: List of clustering assignments
            
        Returns: The Equivalence class of each cluster.
        
        Raises:
            NA 
        """
        ###
        #TODO: alternative is to have two columns for min and max.
        data = self.change_numerical_to_str(self.dataset)
        ###
        for cluster in set(self.cluster_assignments):
            filtered_data = data[self.cluster_assignments == cluster]
            equivalence_class = self.get_lowest_common_ancestor(filtered_data)
            for key, value in equivalence_class.items():
                filtered_data[key] = '{}'.format(value)
            data[self.cluster_assignments == cluster] = filtered_data
    
        #as an option for the future, to append an extra column holding
        #the cluster assignments
        newdata = rfn.append_fields(data, 'cluster', self.cluster_assignments).data
        return newdata
    
    def get_lowest_common_ancestor(self, data):
        """ Computes the lowest common ancestor (LCA) for the given dataset.
        
        Description: A function that will compute the LCA for the given dataset
        
        Args: 
            data: Structured Numpy Array for which the LCA is desired.
            features: List of attributes to be used when clustering.
            lca_funcs: List of user provided functions - one for each
                        'QI' to be used to compute the lower common ancestor
                        for a group of points.
            
        Returns: The lowest common ancestor of the dataset, across all features.
        
        Raises:
            NA 
            """
        equivalence_class = {}
        for attr_name, attr_lca_func in self.lca_funcs.items():
            if attr_name in self.quasi_identifiers:
                equivalence_class[attr_name] = attr_lca_func(data[attr_name])
    
        return equivalence_class

    def clustering(self, parameter):
        """ Cluster for l-diversity.
        
        Description: Function that will cluster the data in clusters where 
                    each cluster has at least l distinct entries in the 
                    sensitive_attribute. The algorithm is an extension
                    of the algorithm presented in the paper 
                    'Efficient k-Anonymization Using Clustering Techniques'
                    that deals with k-anonymity.
        
        Args: 
            data: Structured Numpy Array dataset to be anonymized.
            sensitive_attribute: String for the name of the attribute to be 
                                protected
            l: The minimum number of distinct entries in each cluster,
                for the sensitive_attribute
            
        Returns: Clustering assignments for the data provided.
        
        Raises:
            NA 
        """    
        self.cluster_assignments = self.initialise_clustering()
        
        positions_of_unclustered = np.where(self.cluster_assignments == -1)[0]
        cluster_counter = 0
        
        # The algorithm will first form clusters where each one has at least
        # l distinct entries in the sensitive_attribute feature, and then
        # assign the extras to the existing clusters.
        
        # A cluster assignment of -1 implies not assigned to a cluster yet.
        while len(set(self.SA[positions_of_unclustered])) >= parameter:
            while sum(self.cluster_assignments == cluster_counter) < parameter:
                scores = []
                positions_of_cluster = np.where(self.cluster_assignments == cluster_counter)[0]
                filt_attribute = self.SA[positions_of_cluster]
                for idx, val in enumerate(self.cluster_assignments):
                    if (val == -1 and self.SA[idx] not in set(filt_attribute)):
                        self.cluster_assignments[idx] = cluster_counter
                        scores.append((idx, self.get_total_information_loss(self.dataset)))
                        self.cluster_assignments[idx] = -1
                best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
                self.cluster_assignments[best] = cluster_counter
    
            cluster_counter += 1
            positions_of_unclustered = np.where(self.cluster_assignments == -1)[0]
            
        self.assign_extras()
        
    def check_input(self):
        """ Cheking whether the input is right.
        
        Description: Checks the input.
        
        Args: ...
        
        Raises:
            
        """
                   
        if type(self.dataset) is not np.ndarray:
            raise TypeError('data needs to be of type "np.ndarray" ')
            
        if type(self.quasi_identifiers) is not list:
            raise TypeError('quasi_identifiers needs to be of type "list" ')
        
        if type(self.sensitive_attributes) is not list:
            raise TypeError('sensitive_attributes needs to be of type "list" ')
            
        if type(self.attributes_to_suppress) is not list:
            raise TypeError('attributes_to_suppress needs to be of type "list" ')
    
        for item in self.quasi_identifiers:
            if type(item) is not str:
                raise TypeError('quasi_identifiers should be of type "string" ')
        
        for item in self.sensitive_attributes:
            if type(item) is not str:
                raise TypeError('sensitive_attributes should be of type "string" ')
                
        for item in self.attributes_to_suppress:
            if type(item) is not str:
                raise TypeError('attributes_to_suppress should be of type "string" ')
                
        if not self.sensitive_attributes:
            raise ValueError('No Sensitive Attributes Provided')
        
        if not self.quasi_identifiers:
            raise ValueError('No Quasi-Identifiers Provided')

        attributes = self.features
        for attr in self.sensitive_attributes:
            if attr not in attributes:
                raise NameError(str(attr) + ' in Sensitive Attributes, not found in the dataset')
        
        for attr in self.quasi_identifiers:
            if attr not in attributes:
                raise NameError(str(attr) + ' in Quasi-Identifiers, not found in the dataset')
            
        for attr in self.attributes_to_suppress:
            if attr not in attributes:
                raise NameError(str(attr) + ' in attributes to suppresss, not found in the dataset')
 
class KAnonymity(BaseAnonymiser):
    def __init__(self, 
                 dataset,
                 identifiers,
                 quasi_identifiers,
                 sensitive_attributes,
                 lca_funcs,
                 range_funcs,
                 k=None):
        super().__init__(
                         dataset,
                         identifiers,
                         quasi_identifiers,
                         sensitive_attributes,
                         lca_funcs,
                         range_funcs)
        self.k = k
        
    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, k):
        if k <= 0:
            raise ValueError("k needs to be positive")
        elif k > self.n_samples:
            raise ValueError("k needs to be smaller than the size of the dataset")
        else:
            self._k = k
            
    def clustering(self):
        """ Cluster for k-anonymity.
        
        Description: Function that will cluster the data in clusters of 
                    size at least k. The algorithm follows from paper 
                    'Efficient k-Anonymization Using Clustering Techniques'.
        
        Args: 
            data: Structured Numpy Array dataset to be anonymized.
            features: List of attributes to be used when clustering.
            range_funcs: List of user provided functions - one for each 'QI' 
                        to be used to calculate the range of a feature in
                        a group of points. For continuous variables, it is the 
                        range of the variable in the group (MAX - MIN) and for 
                        categorical it is the height of the tree.
            k: The minimum size of the equivalence classes.
            
        Returns: Clustering assignments for the data provided, where each
                cluster is at least of size k.
        
        Raises:
            NA 
        """
        self.cluster_assignments = self.initialise_clustering()
        cluster_counter = 0
        # The algorithm will first form clusters of size k with an objective,
        # of minimizing Information Loss, and then assign the leftover to the 
        # existing clusters.
        
        # A cluster assignment of -1 implies not assigned to a cluster yet.
        while sum(self.cluster_assignments == -1) > self.k:
            while sum(self.cluster_assignments == cluster_counter) < self.k:
                scores = []
                for idx, val in enumerate(self.cluster_assignments):
                    if val == -1:
                        self.cluster_assignments[idx] = cluster_counter
                        scores.append((idx, 
                                       self.get_total_information_loss(self.dataset)))
                        self.cluster_assignments[idx] = -1
                best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
                self.cluster_assignments[best] = cluster_counter
            cluster_counter += 1
         
        # Stop forming new clusters, since now we have fewer than k elements left,
        # and start adding them to existing clusters, by trying to minimize
        # information loss again.
        self.assign_extras()
    
    
    def apply_kanonymity(self, 
                         suppress=False,
                         k=None):
        if k is None:
            if self.k is None:
                raise ValueError("no k provided")
        else:
            self.k = k

        if suppress:
            self.suppress()
        self.clustering()
        data = self.get_equivalence_classes()
        
        return data
    

class LDiversity(BaseAnonymiser):
    def __init__(self, 
                 dataset,
                 identifiers,
                 quasi_identifiers,
                 sensitive_attributes,
                 lca_funcs,
                 range_funcs,
                 l=None):
        super().__init__(
                           dataset,
                           identifiers,
                           quasi_identifiers,
                           sensitive_attributes,
                           lca_funcs,
                           range_funcs)
        self.l = l
    
    @property
    def l(self):
        return self._l
    
    @l.setter
    def l(self, l):
        if l <= 0:
            raise ValueError("l needs to be positive")
        elif l > len(set(self.SA)):
            raise ValueError("l needs to be smaller than the number of features")
        else:
            self._l = l
            
    def apply_ldiversity(self, 
                         suppress=False,
                         l=None):
        if l is None:
            if self.l is None:
                raise ValueError("no l provided")
        else:
            self.l = l

        if suppress:
            self.suppress()
        self.clustering(self.l)
        data = self.get_equivalence_classes()
        
        return data
    
class TCloseness(BaseAnonymiser):
    def __init__(self, 
             dataset,
             identifiers,
             quasi_identifiers,
             sensitive_attributes,
             lca_funcs = {},
             range_funcs = {},
             t=None):
        super().__init__(
                           dataset,
                           identifiers,
                           quasi_identifiers,
                           sensitive_attributes,
                           lca_funcs,
                           range_funcs)
        self.t = t
    
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        if t <= 0:
            raise ValueError("k needs to be positive")
        else:
            self._t = t
            
    def merge_clusters(self, clusters_tobe_merged):
        newcluster = int(np.max(self.cluster_assignments) + 1)
        for idx, val in enumerate(self.cluster_assignments):
            if int(val) in list(map(int, clusters_tobe_merged)):
                self.cluster_assignments[idx] = newcluster

    def check_tcloseness(self):
        """
        Check whether the t-closeness condition is satisfied, for all the clusters.
        """
        satisfied = True
        emds = []
        for cluster in set(self.cluster_assignments):
            filtered_attribute = self.SA[np.where(self.cluster_assignments == cluster)[0]]
            emd = get_emd_fordistrs(get_distr(filtered_attribute), self.base_distr)
            emds.append((cluster, emd))
            if emd >= self.t:
                satisfied = False
        return satisfied, emds

    def generalize(self):
        #self.base_counts = Counter(self.dataset[self.sensitive_attributes[0]])
        #unique, counts = np.unique(self.SA, return_counts=True)
        #self.base_counts = dict(zip(unique, counts))
        self.base_counts = self.SA
        self.base_distr = get_distr(self.base_counts)
    
        # If the formed clusters satisfy the t-closeness condition,
        # then proceed to form the equivalence classes.
        satisfied, emds = self.check_tcloseness()
        
        # While the t-closeness condition has not been met, start merging clusters
        # until it is satisfied.
        # ** EMD(P, Q) <= max{EMD(P0, Q), EMD(P1, Q)} **
        # where P0 and P1 are the distributions of the sensitive attribute in two
        # clusters and P is the distribution of sensitive attribute when the
        # two clusters are merged. Q is the distribution of the sensitive attribute
        # on the full dataset.
        
        # The procedure is to find the cluster that has the highest EMD to Q
        # and then find a second cluster which is the closest to the first
        # and merge them.
        while not satisfied:

            clusters = [item[0] for item in emds]
            furthest_cluster = clusters[np.argmax([item[1] for item in emds])]
            pos = np.where(self.cluster_assignments == furthest_cluster)[0]
            list1 = self.SA[pos]
            scores = []
            for cluster in set(self.cluster_assignments):
                if cluster == furthest_cluster:
                    scores.append((cluster, 1000))
                    continue
                list2 = self.SA[np.where(self.cluster_assignments == cluster)[0]]
                emd = get_emd_forlists(list1, list2)
                scores.append((cluster, emd))
            emds_tocluster = np.array([item[1] for item in scores])
            closest_cluster = clusters[emds_tocluster.argsort()[0]]
            clusters_tobe_merged = [furthest_cluster, closest_cluster]
            self.merge_clusters(clusters_tobe_merged)
            satisfied, emds = self.check_tcloseness()
           
    def apply_tcloseness(self, 
                         suppress=False,
                         t=None):
        if t is None:
            if self.t is None:
                raise ValueError("no t provided")
        else:
            self.t = t

        if suppress:
            self.suppress()
        starting_size = len(set(self.SA))
        self.clustering(starting_size)
        self.generalize()
        data = self.get_equivalence_classes()
        
        return data
    
    def swap_instances(self):
        """ Swaps instances between a cluster and the pool of unlabelled instances.
        
        Description: Find the best possible swap of instances between the given
                    cluster and the pool of unlabelled instances. It first selects 
                    the instance who reduces the information loss within the cluster
                    and then finds the best replacement with respect to reducing
                    the EMD to the base distribution.
                       
        Args: 
            data: Structured Numpy Array dataset.
            sensitive_attribute: String for the name of the attribute to be
                                protected.
            base_distr: The distribution of the sensitive_attribute in the whole
                        dataset.
            cluster_assignments: List of cluster assignments
            t: The maximum allowed Earth Mover's Distance (EMD) distance allowed
                between each equivalence class' distribution of sensitive_attribute
                and the overall distribution of the sensitive_attribute.
            
        Returns: cluster_assignments: List with updated cluster_assignments, with a 
                                    lower EMD to the base distribution.
        
        Raises:
            NA 
            """
        cluster_counter = int(max(self.cluster_assignments))
        scores = []
        
        full_dataset_ranges = self.get_feature_ranges(self.dataset)
        
        for idx, val in enumerate(self.cluster_assignments):
            if val == cluster_counter:
                self.cluster_assignments[idx] = -1
                filtered_dataset = self.filter_dataset(self.dataset, [cluster_counter])
                scores.append((idx, 
                               self.get_information_loss(filtered_dataset,
                                                         full_dataset_ranges)))
                self.cluster_assignments[idx] = cluster_counter
                
        best_internal = scores[np.argmax([item[1] for item in scores])][0]
        
        scores = []
        for external_idx, external_val in enumerate(self.cluster_assignments):
            if external_val == -1:
                self.cluster_assignments[best_internal] = -1
                self.cluster_assignments[external_idx] = cluster_counter
                new_distr = get_distr(self.SA[np.where(self.cluster_assignments == cluster_counter)[0]])
                new_emd = get_emd_fordistrs(new_distr, self.base_distr)
                
                scores.append((external_idx, new_emd))
                
                self.cluster_assignments[best_internal] = cluster_counter
                self.cluster_assignments[external_idx] = -1
        best = np.argmin([item[1] for item in scores])
        best_external = scores[best][0]
        
        self.cluster_assignments[best_internal] = -1
        self.cluster_assignments[best_external] = cluster_counter

    def apply_tcloseness_2(self, 
                           suppress=False,
                           t=None):
        """ Cluster and generalize for t-diversity.
        
        Description: Function that will call clustering and generalization
                    functions for t-closeness. The algorithm will first form
                    one cluster and then keep swapping instances with the
                    set of unlabelled instances until the first cluster satisfies
                    t-closeness. It then continues by forming the second cluster
                    and so on and so forth.
                    
                    Does not guarantee t-closeness in its current form.
        
        Args: 
            data: Structured Numpy Array dataset to be anonymized.
            features: List of attributes to be used when clustering.
            lca_funcs: List of user provided functions - one for each
                        'QI' to be used to compute the lower common ancestor
                        for a group of points.
            range_funcs: List of user provided functions - one for each 'QI' 
                        to be used to calculate the range of a feature in
                        a group of points. For continuous variables, it is the 
                        range of the variable in the group (MAX - MIN) and for 
                        categorical it is the height of the tree.
            sensitive_attribute: The name (string) of the attribute to be protected.
            t: The minimum allowed distance between the distribution of the 
                sensitive attribute in each equivalence class and the distribution
                of the attribute in the full dataset, as calculated by the 
                Earth Movers Distance (EMD).
            
        Returns: A dataset that satisfies t-closeness.
        
        Raises:
            NA 
            """
            
        if t == None:
            if self.t == None:
                raise ValueError("no t provided")
        else:
            self.t = t

        if suppress:
            self.suppress()

        self.base_counts = self.SA
        self.base_distr = get_distr(self.base_counts)
        
        starting_size = len(set(self.SA))
        
        self.cluster_assignments = self.initialise_clustering()
        
        cluster_counter = 0
        positions_of_unclustered = np.where(self.cluster_assignments == -1)[0]
    
        while len(positions_of_unclustered) > starting_size:
            while len(np.where(self.cluster_assignments == cluster_counter)[0]) < starting_size:
                scores = []
                for idx, val in enumerate(self.cluster_assignments):
                    if val == -1:
                        self.cluster_assignments[idx] = cluster_counter
                        scores.append((idx, 
                                       self.get_total_information_loss(self.dataset)))
                        self.cluster_assignments[idx] = -1
                best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
                self.cluster_assignments[best] = cluster_counter
            new_distr = get_distr(self.SA[np.where(self.cluster_assignments == cluster_counter)[0]])    
            emd = get_emd_fordistrs(new_distr, self.base_distr)
            satisfied = emd <= self.t
    
            if not satisfied:
                self.swap_instances()
                #cluster_assignments = swap_instances_(SA, base_distr, cluster_assignments, cluster_counter, t)
            positions_of_unclustered = np.where(self.cluster_assignments == -1)[0]
            cluster_counter += 1
            
        satisfied, emds = self.check_tcloseness()
        self.assign_extras()
    
        data = self.get_equivalence_classes()
        satisfied, emds = self.check_tcloseness()
    
        return data
    

    