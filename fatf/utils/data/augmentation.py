import collections

def lime(num_samples,
         data_set,
         data_row=None):
    """Generates a neighborhood around a prediction.

    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to
    the means and stds in the training data. For categorical features,
    perturb by sampling according to the training distribution, and making
    a binary feature that is 1 when the value is the same as the instance
    being explained.

    Args:
        data_row: 1d numpy array, corresponding to a row
        num_samples: size of the neighborhood to learn the linear model

    Returns:
        A tuple (data, inverse), where:
            data: dense num_samples * K matrix, where categorical features
            are encoded with either 0 (not equal to the corresponding value
            in data_row) or 1. The first row is the original instance.
            inverse: same as data, except the categorical features are not
            binary, but categorical (as the original data)
    """

    # TODO: Check input array is 2D; data row is 1D
    # TODO: num samples has to be a positive integer

    # TODO
    categorical_features, numerical_features = TODO_get_feature_types()

    features_mean = np.mean(data_set, axis=1)
    features_var = np.var(data_set, axis=1)
    features_std = np.sqrt(features_var)

    feature_frequencies = dict()
    feature_values = dict()
    for cf in categorical_features:
        feature_vector = data_set[cf]
        feature_counter = collections.Counter(feature_vector)

        feature_values[cf] = list(feature_counter.keys())

        feature_frequencies[cf] = np.array(list(feature_counter.values()))
        feature_frequencies[cf] /= np.sum(feature_frequencies[cf])

    # If data point is not given take a global sample (based on the mean of all features)
    data_row = features_mean if data_row is None else data_row

    num_features = data_set.shape[0]
    data = np.zeros((num_samples, num_features))

    data = np.random.normal(
            0, 1, num_samples * num_features).reshape(
            num_samples, num_features)
    data = data * features_std + data_row

    data[0] = data_row.copy()
    for column in categorical_features:
        vals = feature_values[column]
        freqs = feature_frequencies[column]

        new_column = np.random.choice(vals, size=num_samples,
                                          replace=True, p=freqs)
        new_column[0] = data[0, column]

        data[:, column] = new_column
    data[0] = data_row.copy()

    return data


def lime_data(generated_data, original_data):
    binary_data = generated_data.copy()
    for column in categorical_features:
        binary_column = np.array([1 if x == data_row[column]
                                  else 0 for x in inverse_column])
        binary_column[0] = 1
        data[:, column] = binary_column
    pass
