def check_systematic_error(self,
                           predictions: np.ndarray,
                           requested_checks: Optional[List[str]] = None,
                           features_to_check: Optional[List[str]] = None,
                           boundaries_for_numerical: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """ Checks for systematic error in the dataset.

    Will check if the different sub-populations defined by the
    features_to_check have similar behaviour under the model.

    Parameters
    ----------
    predictions: np.ndarray
        Predictionas of a model, built using the data provided.
    requested_checks: list of strings
        Corresponding to which checks to perform
    features_to_check: List of Strings
        for which features to consider the sub-populations.
    boundaries_for_numerical: Dict of List of tuples
        defining the bins of numerical data.

    Returns
    -------
    summary: dict
        Dictionary of confusion matrices for each sub-population,
        if checks==None, else, Dictionary of Dictionaries,
        one for each sub-population.

        """
    if not boundaries_for_numerical:
        boundaries_for_numerical = {}
    self.predictions = predictions
    multiclass = False
    classes_list = list(set(self.targets))
    if len(classes_list) > 2:
        multiclass = True

    if not requested_checks:
        if multiclass:
            requested_checks = list(self.checks_multiclass.keys())
        else:
            requested_checks = list(self.checks.keys())

    if not features_to_check:
        if self.features_to_check is None:
            raise ValueError('no features to check provided')
    else:
        self.features_to_check = features_to_check
    cross_product = self._get_cross_product(boundaries_for_numerical)
    summary = {}
    for combination in cross_product:
        filtered_predictions, filtered_targets = \
            self._apply_combination_filter(combination, boundaries_for_numerical)
        conf_mat = _get_confusion_matrix(filtered_targets,
                                              filtered_predictions,
                                              classes_list)
        if not requested_checks:
            summary[combination] = conf_mat
        else:
            summary[combination] = {}
            if multiclass:
                for idx, target_class in enumerate(classes_list):
                    summary[combination][target_class] = {}
                    for item in requested_checks:
                        summary[combination][target_class][item] = \
                            self.checks_multiclass[item](conf_mat, idx)
            else:
                for item in requested_checks:
                    summary[combination][item] = self.checks[item](conf_mat)
    return summary


def check_sampling_bias(self,
                        features_to_check: Optional[List[str]] = None,
                        return_weights: Optional[bool] = False,
                        boundaries_for_numerical: Optional[Dict[str, np.ndarray]] = None) -> Union[Tuple[dict, np.ndarray], dict]:
    """ Checks for sampling bias in the dataset.

    Will check if the different sub-populations defined by the
    features_to_check have similar representation (sample size).

    Parameters
    ----------
    features_to_check: List of Strings
        for which features to consider the sub-populations.
    return_weights: Boolean
        on whether to return weights to be used for cost-sensitive learning.
    boundaries_for_numerical: Dict of List of tuples
        defining the bins of numerical data.

    Returns
    -------
        counts: dict
            of data for each sub-population defined by the cross-product
            of the provided features.
        weights: Optional, np.ndarray
            weights to be used for cost-sensitive learning.

        """
    if not boundaries_for_numerical:
        boundaries_for_numerical = {}

    if not features_to_check:
        if self.features_to_check is None:
            raise ValueError('no features to check provided')
    else:
        self.features_to_check = features_to_check
    counts: dict = {}
    cross_product = self._get_cross_product(boundaries_for_numerical)
    counts = self._get_counts(cross_product, boundaries_for_numerical)
    if not return_weights:
        return counts
    else:
        weights = self._get_weights_costsensitivelearning(counts, boundaries_for_numerical)
        return counts, weights


def check_systemic_bias(self,
                        threshold: float = 0.1) -> list:
    """ Checks for systemic bias in the dataset.

    Will check if similar instances, that differ only on the
    protected attribute have been treated differently. Treated
    refers to the 'target' of the instance.

    Parameters
    ----------
    threshold: Float
        value for what counts as similar. Default to 0.1.

    Returns
    -------
    distance_list: list
        List of pairs of instances that are similar but have been treated
            differently.

    """
    n_samples = self.dataset.shape[0]
    if self.structured_bool:
        protected = self.dataset[self.protected_field]
    else:
        protected = self.dataset[:, self.protected_field].astype(int)
    distance_list = []
    for i in range(n_samples):
        v0 = self.dataset[i]
        protected0 = protected[i]
        target0 = self.targets[i]
        for j in range(i):
            v1 = self.dataset[j]
            protected1 = protected[j]
            target1 = self.targets[j]
            dist = self._apply_distance_funcs(v0, v1, toignore=[self.protected_field])

            same_protected = protected0 == protected1
            same_target = target0 == target1
            if (dist <= threshold and
                same_protected == False and
                same_target == False):
                distance_list.append((dist, (i,j)))
    return distance_list
