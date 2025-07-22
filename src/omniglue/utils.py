import tensorflow as tf


def soft_assignment_to_match_matrix(soft_assignment: tf.Tensor, match_threshold: float) -> tf.Tensor:
    """Converts a matrix of soft assignment values to binary yes/no match matrix.

    Note: implementation based on https://github.com/google-research/omniglue/blob/main/src/omniglue/omniglue_extract.py.

    Searches soft_assignment for row- and column-maximum values, which indicate
    mutual nearest neighbor matches between two unique sets of keypoints. Also,
    ensures that score values for matches are above the specified threshold.

    :param soft_assignment: (B, N, M) tensor, contains matching likelihood value between features of
    different sets. N is number of features in image0, and M is number of features in image1. Higher
    value indicates more likely to match.
    :type soft_assignment: tf.Tensor
    :param match_threshold: thresholding value to consider a match valid.
    :type match_threshold: float
    :return: (B, N, M) tensor of binary values. A value of 1 at index (x, y) indicates a match between
    index 'x' (out of N) in image0 and index 'y' (out of M) in image 1.
    :rtype: tf.Tensor
    """

    def _range_like(x, dim):
        """Returns tensor with values (0, 1, 2, ..., N) for dimension in input x."""
        return tf.range(tf.shape(x)[dim], dtype=x.dtype)

    matches = tf.TensorArray(tf.float32, size=tf.shape(soft_assignment)[0])
    for i in range(tf.shape(soft_assignment)[0]):
        # Iterate through batch and process one example at a time.
        scores = tf.expand_dims(soft_assignment[i, :], 0)  # Shape: (1, N, M).

        # Find indices for max values per row and per column.
        max0 = tf.math.reduce_max(scores, axis=2)  # Shape: (1, N).
        indices0 = tf.math.argmax(scores, axis=2)  # Shape: (1, N).
        indices1 = tf.math.argmax(scores, axis=1)  # Shape: (1, M).

        # Find matches from mutual argmax indices of each set of keypoints.
        mutual = tf.expand_dims(_range_like(indices0, 1), 0) == tf.gather(indices1, indices0, axis=1)

        # Create match matrix from sets of index pairs and values.
        kp_ind_pairs = tf.stack([_range_like(indices0, 1), tf.squeeze(indices0)], axis=1)
        mutual_max0 = tf.squeeze(tf.squeeze(tf.where(mutual, max0, 0), 0))
        sparse = tf.sparse.SparseTensor(kp_ind_pairs, mutual_max0, tf.shape(scores, out_type=tf.int64)[1:])
        match_matrix = tf.sparse.to_dense(sparse)
        matches = matches.write(i, match_matrix)

    # Threshold on match_threshold value and convert to binary (0, 1) values.
    match_matrix = matches.stack()
    match_matrix = match_matrix > match_threshold

    return match_matrix
