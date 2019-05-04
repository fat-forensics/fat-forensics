def counterfactual_fairness(test_model: Any,
                            protected: str,
                            xtest_: np.ndarray,
                            unique_targets: List[Any]) -> np.matrix:
    """ Checks counterfactual fairness of the model.

    Will flip the protected attribute and generate new predictions,
    to check the model's dependence on the protected feature.

    Parameters
    ----------
    model: Object
        *trained* model to be used for generating predictions
    protected: str
        name of protected field
    xtest_: np.ndarray
        containing the test data.
    unique_targets: list
        containing the unique targets, to be used to order the
        confusion matrix.

    Returns
    -------
    conf_mat: np.matrix
        Confusion matrix between the predictions before and after the
            flipping the protected feature

    Raises
    ------
    TypeError
        If the model provided does not have the necessary functionality.
        """
    is_functional = fumv.check_model_functionality(test_model)
    if not is_functional:
        raise TypeError('Model provided is not proper')

    xtest = xtest_.copy(order='K')
    original_predictions = test_model.predict(np.array(xtest.tolist()))
    xtest[protected] = [int(not item) for item in xtest[protected]]
    modified_X = np.array(xtest.tolist())
    counterfactual_predicitons = test_model.predict(modified_X)
    conf_mat = _get_confusion_matrix(original_predictions,
                                          counterfactual_predicitons,
                                          unique_targets)
    return conf_mat

def individual_fairness(model: Any,
                        X: np.ndarray,
                        X_distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                        predictions_distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> bool:
    """ Checks individual fairness -- 'Fairness through awareness'.

    Will check whether similar instances get similar predictions.

    Parameters
    ----------
    model: Object
        *trained* model to be used for generating predictions
    X: np.ndarray
        containing the design matrix.
    X_distance_func: Function
        to be used to compute distance between instances.
    predictions_distance_func: Function
        to be used to compute distance between predictions.

    Returns
    -------
    bool
        Will check whether 'fairness through awareness holds'
            d_X(x_i, x_j) <= d_f(f(x_i), f(x_j))

    Raises
    ------
    TypeError
        If the model provided does not have necessary functionality.
        """
    is_functional = fumv.check_model_functionality(model)
    if not is_functional:
        raise TypeError('Model provided is not proper')

    if not X_distance_func:
        X_distance_func = euc_dist
    if not predictions_distance_func:
        predictions_distance_func = euc_dist

    n = X.shape[0]
    X_distance_mat = get_distance_mat(X, X_distance_func)
    predictions_proba = model.predict_proba(X)
    y_distance_mat = get_distance_mat(predictions_proba,
                                      predictions_distance_func)
    for i in range(n):
        for j in range(i):
            if y_distance_mat[i, j] > X_distance_mat[i, j]:
                return False
    return True
