from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from torch import nn

from .utils import soft_assignment_to_match_matrix

MODEL_PATH = Path(__file__).parents[2].joinpath("models", "og_export")


class FlexibleOmniGlue(nn.Module):
    """FlexibleOmniGlue sparse and dense feature matching using attention."""

    def __init__(self, matching_threshold: float = 1e-3) -> None:
        """Constructor of FlexibleOmniGlue class.

        :param matching_threshold: The matching threshold value to consider if a match is valid, defaults to 1e-3.
        :type matching_threshold: float, optional
        """

        super().__init__()

        self.matching_threshold = matching_threshold
        self.matcher = None

    def load(self, p: Path | str = MODEL_PATH) -> None:
        """Load the model.

        :param path: The path to the tensorflow extract, defaults to MODEL_PATH.
        :type path: Path | str, optional
        """

        self.matcher = tf.saved_model.load(p)

    def forward(
        self,
        img0_shape_rc: tuple[int, int],
        img1_shape_rc: tuple[int, int],
        img0_sparse_keypoints: np.ndarray,
        img0_sparse_descriptors: np.ndarray,
        img1_sparse_keypoints: np.ndarray,
        img1_sparse_descriptors: np.ndarray,
        img0_dense_descriptors: np.ndarray,
        img1_dense_descriptors: np.ndarray,
        img0_sparse_score: Optional[np.ndarray] = None,
        img1_sparse_score: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match the keypoints from the two images based on their sparse and dense features.

        :param img0_shape_rc: The dimension of the first image as (height, width) or (row, column).
        :type img0_shape_rc: tuple[int, int]
        :param img1_shape_rc: The dimension of the second image as (height, width) or (row, column).
        :type img1_shape_rc: tuple[int, int]
        :param img0_sparse_keypoints: The sparse keypoints of the first image as (x, y).
        :type img0_sparse_keypoints: np.ndarray
        :param img0_sparse_descriptors: The sparse descriptors of the first image.
        :type img0_sparse_descriptors: np.ndarray
        :param img1_sparse_keypoints: The sparse keypoints of the second image as (x, y).
        :type img1_sparse_keypoints: np.ndarray
        :param img1_sparse_descriptors: The sparse descriptors of the second image.
        :type img1_sparse_descriptors: np.ndarray
        :param img0_dense_descriptors: The dense descriptors of the first image.
        :type img0_dense_descriptors: np.ndarray
        :param img1_dense_descriptors: The dense descriptors of the second image.
        :type img1_dense_descriptors: np.ndarray
        :param img0_sparse_score: The score associated with the sparse keypoints of the first image, defaults to None.
        :type img0_sparse_score: Optional[np.ndarray], optional
        :param img1_sparse_score: The score associated with the sparse keypoints of the second image, defaults to None.
        :type img1_sparse_score: Optional[np.ndarray], optional
        :return: The matches indices as [img0_idx, corresponding_img1_idx] and the confidences of each match.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Dummy score if not provided
        if img0_sparse_score is None or img1_sparse_score is None:
            img0_sparse_score = np.ones((img0_sparse_keypoints.shape[0]))
            img1_sparse_score = np.ones((img1_sparse_keypoints.shape[0]))

        # Format the input as the model expects them
        inputs = self._construct_inputs(
            img0_shape_rc,
            img1_shape_rc,
            img0_sparse_keypoints,
            img0_sparse_descriptors,
            img0_sparse_score,
            img1_sparse_keypoints,
            img1_sparse_descriptors,
            img1_sparse_score,
            img0_dense_descriptors,
            img1_dense_descriptors,
        )

        assert self.matcher is not None

        og_outputs = self.matcher.signatures["serving_default"](**inputs)
        soft_assignment = og_outputs["soft_assignment"][:, :-1, :-1]

        match_matrix = soft_assignment_to_match_matrix(soft_assignment, self.matching_threshold).numpy().squeeze()

        # Filter out any matches with 0.0 confidence keypoints.
        match_indices = np.argwhere(match_matrix)
        keep = []
        for i in range(match_indices.shape[0]):
            m = match_indices[i, :]
            if (img0_sparse_score[m[0]] > 0.0) and (img1_sparse_score[m[1]] > 0.0):
                keep.append(i)
        match_indices = match_indices[keep]
        match_confidences = np.array([soft_assignment[0, m[0], m[1]] for m in match_indices])

        return match_indices, match_confidences

    def __call__(
        self,
        img0_shape_rc: tuple[int, int],
        img1_shape_rc: tuple[int, int],
        img0_sparse_keypoints: np.ndarray,
        img0_sparse_descriptors: np.ndarray,
        img1_sparse_keypoints: np.ndarray,
        img1_sparse_descriptors: np.ndarray,
        img0_dense_descriptors: np.ndarray,
        img1_dense_descriptors: np.ndarray,
        img0_sparse_score: Optional[np.ndarray] = None,
        img1_sparse_score: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match the keypoints from the two images based on their sparse and dense features.

        :param img0_shape_rc: The dimension of the first image as (height, width) or (row, column).
        :type img0_shape_rc: tuple[int, int]
        :param img1_shape_rc: The dimension of the second image as (height, width) or (row, column).
        :type img1_shape_rc: tuple[int, int]
        :param img0_sparse_keypoints: The sparse keypoints of the first image as (x, y).
        :type img0_sparse_keypoints: np.ndarray
        :param img0_sparse_descriptors: The sparse descriptors of the first image.
        :type img0_sparse_descriptors: np.ndarray
        :param img1_sparse_keypoints: The sparse keypoints of the second image as (x, y).
        :type img1_sparse_keypoints: np.ndarray
        :param img1_sparse_descriptors: The sparse descriptors of the second image.
        :type img1_sparse_descriptors: np.ndarray
        :param img0_dense_descriptors: The dense descriptors of the first image.
        :type img0_dense_descriptors: np.ndarray
        :param img1_dense_descriptors: The dense descriptors of the second image.
        :type img1_dense_descriptors: np.ndarray
        :param img0_sparse_score: The score associated with the sparse keypoints of the first image, defaults to None.
        :type img0_sparse_score: Optional[np.ndarray], optional
        :param img1_sparse_score: The score associated with the sparse keypoints of the second image, defaults to None.
        :type img1_sparse_score: Optional[np.ndarray], optional
        :return: The matches indices as [img0_idx, corresponding_img1_idx] and the confidences of each match.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        return self.forward(
            img0_shape_rc,
            img1_shape_rc,
            img0_sparse_keypoints,
            img0_sparse_descriptors,
            img1_sparse_keypoints,
            img1_sparse_descriptors,
            img0_dense_descriptors,
            img1_dense_descriptors,
            img0_sparse_score,
            img1_sparse_score,
        )

    def FindMatches(
        self,
        img0_shape_rc: tuple[int, int],
        img1_shape_rc: tuple[int, int],
        img0_sparse_keypoints: np.ndarray,
        img0_sparse_descriptors: np.ndarray,
        img1_sparse_keypoints: np.ndarray,
        img1_sparse_descriptors: np.ndarray,
        img0_dense_descriptors: tf.Tensor,
        img1_dense_descriptors: tf.Tensor,
        img0_sparse_score: Optional[np.ndarray] = None,
        img1_sparse_score: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match the keypoints from the two images based on their sparse and dense features.

        Note: implementation based on https://github.com/google-research/omniglue/blob/main/src/omniglue/omniglue_extract.py.

        :param img0_shape_rc: The dimension of the first image as (height, width) or (row, column).
        :type img0_shape_rc: tuple[int, int]
        :param img1_shape_rc: The dimension of the second image as (height, width) or (row, column).
        :type img1_shape_rc: tuple[int, int]
        :param img0_sparse_keypoints: The sparse keypoints of the first image as (x, y).
        :type img0_sparse_keypoints: np.ndarray
        :param img0_sparse_descriptors: The sparse descriptors of the first image.
        :type img0_sparse_descriptors: np.ndarray
        :param img1_sparse_keypoints: The sparse keypoints of the second image as (x, y).
        :type img1_sparse_keypoints: np.ndarray
        :param img1_sparse_descriptors: The sparse descriptors of the second image.
        :type img1_sparse_descriptors: np.ndarray
        :param img0_dense_descriptors: The dense descriptors of the first image.
        :type img0_dense_descriptors: tf.Tensor
        :param img1_dense_descriptors: The dense descriptors of the second image.
        :type img1_dense_descriptors: tf.Tensor
        :param img0_sparse_score: The score associated with the sparse keypoints of the first image, defaults to None.
        :type img0_sparse_score: Optional[np.ndarray], optional
        :param img1_sparse_score: The score associated with the sparse keypoints of the second image, defaults to None.
        :type img1_sparse_score: Optional[np.ndarray], optional
        :return: The matches indices as [img0_idx, corresponding_img1_idx] and the confidences of each match.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        indices, match_confidences = self.forward(
            img0_shape_rc,
            img1_shape_rc,
            img0_sparse_keypoints,
            img0_sparse_descriptors,
            img1_sparse_keypoints,
            img1_sparse_descriptors,
            img0_dense_descriptors.numpy(),
            img1_dense_descriptors.numpy(),
            img0_sparse_score,
            img1_sparse_score,
        )

        return (
            np.array([img0_sparse_keypoints[i[0], :] for i in indices]),
            np.array([img1_sparse_keypoints[i[1], :] for i in indices]),
            match_confidences,
        )

    def _construct_inputs(
        self,
        img0_shape_rc: tuple[int, int],
        img1_shape_rc: tuple[int, int],
        img0_sparse_keypoints: np.ndarray,
        img0_sparse_descriptors: np.ndarray,
        img0_sparse_score: np.ndarray,
        img1_sparse_keypoints: np.ndarray,
        img1_sparse_descriptors: np.ndarray,
        img1_sparse_score: np.ndarray,
        img0_dense_descriptors: np.ndarray,
        img1_dense_descriptors: np.ndarray,
    ) -> dict[str, tf.Tensor]:
        inputs = {
            "keypoints0": tf.convert_to_tensor(
                np.expand_dims(img0_sparse_keypoints, axis=0),
                dtype=tf.float32,
            ),
            "descriptors0": tf.convert_to_tensor(np.expand_dims(img0_sparse_descriptors, axis=0), dtype=tf.float32),
            "keypoints1": tf.convert_to_tensor(np.expand_dims(img1_sparse_keypoints, axis=0), dtype=tf.float32),
            "descriptors1": tf.convert_to_tensor(np.expand_dims(img1_sparse_descriptors, axis=0), dtype=tf.float32),
            "scores0": tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(img0_sparse_score, axis=0), axis=-1),
                dtype=tf.float32,
            ),
            "scores1": tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(img1_sparse_score, axis=0), axis=-1),
                dtype=tf.float32,
            ),
            "descriptors0_dino": tf.convert_to_tensor(
                tf.expand_dims(img0_dense_descriptors, axis=0), dtype=tf.float32
            ),
            "descriptors1_dino": tf.convert_to_tensor(
                tf.expand_dims(img1_dense_descriptors, axis=0), dtype=tf.float32
            ),
            "width0": tf.convert_to_tensor(np.expand_dims(img0_shape_rc[1], axis=0), dtype=tf.int32),
            "height0": tf.convert_to_tensor(np.expand_dims(img0_shape_rc[0], axis=0), dtype=tf.int32),
            "width1": tf.convert_to_tensor(np.expand_dims(img1_shape_rc[1], axis=0), dtype=tf.int32),
            "height1": tf.convert_to_tensor(np.expand_dims(img1_shape_rc[0], axis=0), dtype=tf.int32),
        }

        return inputs
