import math

import numpy as np
import tensorflow as tf


def _preprocess_shape(h_image, w_image, image_size_max=630, h_down_rate=14, w_down_rate=14):

    # Flatten the tensors
    h_image = tf.squeeze(h_image)
    w_image = tf.squeeze(w_image)
    # logging.info(h_image, w_image)

    h_larger_flag = tf.greater(h_image, w_image)
    large_side_image = tf.maximum(h_image, w_image)

    # Function to calculate new dimensions when height is larger
    def resize_h_larger():
        h_image_target = image_size_max
        w_image_target = tf.cast(image_size_max * w_image / h_image, tf.int32)
        return h_image_target, w_image_target

    # Function to calculate new dimensions when width is larger or equal
    def resize_w_larger_or_equal():
        w_image_target = image_size_max
        h_image_target = tf.cast(image_size_max * h_image / w_image, tf.int32)
        return h_image_target, w_image_target

    # Function to keep original dimensions
    def keep_original():
        return h_image, w_image

    h_image_target, w_image_target = tf.cond(
        tf.greater(large_side_image, image_size_max),
        lambda: tf.cond(h_larger_flag, resize_h_larger, resize_w_larger_or_equal),
        keep_original,
    )

    # resize to be divided by patch size
    h = h_image_target // h_down_rate
    w = w_image_target // w_down_rate
    h_resize = h * h_down_rate
    w_resize = w * w_down_rate

    # Expand dimensions
    h_resize = tf.expand_dims(h_resize, 0)
    w_resize = tf.expand_dims(w_resize, 0)

    return h_resize, w_resize


def lookup_descriptor_bilinear(keypoint: np.ndarray, descriptor_map: np.ndarray) -> np.ndarray:
    """Looks up descriptor value for keypoint from a dense descriptor map.

    Note: implementation based on https://github.com/google-research/omniglue/blob/main/src/omniglue/omniglue_extract.py.

    Uses bilinear interpolation to find descriptor value at non-integer
    positions.

    :param keypoint: 2-dim numpy array containing (x, y) keypoint image coordinates.
    :type keypoint: np.ndarray
    :param descriptor_map: (H, W, D) numpy array representing a dense descriptor map.
    :type descriptor_map: np.ndarray
    :raises ValueError: if kepoint position is out of bounds.
    :return: D-dim descriptor value at the input 'keypoint' location.
    :rtype: np.ndarray
    """

    height, width = np.shape(descriptor_map)[:2]
    if keypoint[0] < 0 or keypoint[0] > width or keypoint[1] < 0 or keypoint[1] > height:
        raise ValueError(
            f"Keypoint position ({keypoint[0]}, {keypoint[1]}) is out of descriptor map bounds ({width} w x"
            f" {height} h)."
        )

    x_range = [math.floor(keypoint[0])]
    if not keypoint[0].is_integer() and keypoint[0] < width - 1:
        x_range.append(x_range[0] + 1)
    y_range = [math.floor(keypoint[1])]
    if not keypoint[1].is_integer() and keypoint[1] < height - 1:
        y_range.append(y_range[0] + 1)

    bilinear_descriptor = np.zeros(np.shape(descriptor_map)[2])
    for curr_x in x_range:
        for curr_y in y_range:
            curr_descriptor = descriptor_map[curr_y, curr_x, :]
            bilinear_scalar = (1.0 - abs(keypoint[0] - curr_x)) * (1.0 - abs(keypoint[1] - curr_y))
            bilinear_descriptor += bilinear_scalar * curr_descriptor
    return bilinear_descriptor


def get_dense_descriptors_from_sparse_keypoints(
    dense_features: np.ndarray, kps: np.ndarray, image_dims: tuple[int, int], feat_dim: int = 768
):
    """Get dense descriptors from sparse keypoints.

    Note: implementation based on https://github.com/google-research/omniglue/blob/main/src/omniglue/omniglue_extract.py.

    :param dense_features: Dense features in 1-D.
    :type dense_features: np.ndarray
    :param kps: Sparse keypoint locations, in format (x, y), in pixels, shape (N, 2).
    :type kps: np.ndarray
    :param image_dims: image height and width (h, w).
    :type image_dims: tuple[int, int]
    :param feat_dim: Dense feature channel size, defaults to 768
    :type feat_dim: int, optional
    :return: Interpolated Dense descriptors.
    """

    keypoints = tf.convert_to_tensor(kps, dtype=tf.float32)
    height = tf.convert_to_tensor(image_dims[0], dtype=tf.int32)
    width = tf.convert_to_tensor(image_dims[1], dtype=tf.int32)
    feature_dim = tf.convert_to_tensor(feat_dim, dtype=tf.int32)

    height_1d = tf.reshape(height, [1])
    width_1d = tf.reshape(width, [1])

    height_1d_resized, width_1d_resized = _preprocess_shape(
        height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14
    )

    height_feat = height_1d_resized // 14
    width_feat = width_1d_resized // 14
    feature_dim_1d = tf.reshape(feature_dim, [1])

    size_feature = tf.concat([height_feat, width_feat, feature_dim_1d], axis=0)
    dino_features = tf.reshape(dense_features, size_feature)

    img_size = tf.cast(tf.concat([width_1d, height_1d], axis=0), tf.float32)
    feature_size = tf.cast(tf.concat([width_feat, height_feat], axis=0), tf.float32)

    keypoints_feature = keypoints / tf.expand_dims(img_size, axis=0) * tf.expand_dims(feature_size, axis=0)

    dino_descriptors = []
    for kp in keypoints_feature:
        dino_descriptors.append(lookup_descriptor_bilinear(kp.numpy(), dino_features.numpy()))
    dino_descriptors = tf.convert_to_tensor(np.array(dino_descriptors), dtype=tf.float32)
    return dino_descriptors


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
