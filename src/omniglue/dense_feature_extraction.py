import math
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
import torch
from torch._prims_common import DeviceLikeType
from typing_extensions import Self


class PatchEmbed(Protocol):
    @property
    def patch_size(self) -> tuple[int, int]: ...


class FoundationModelViT(Protocol):
    @property
    def patch_embed(self) -> PatchEmbed: ...

    def to(self, device: Optional[DeviceLikeType]) -> Self: ...

    def eval(self) -> Self: ...

    def get_intermediate_layers(
        self, x: torch.Tensor, n: int | Sequence
    ) -> tuple[torch.Tensor, ...] | tuple[torch.Tensor | tuple[torch.Tensor]]: ...


class DenseExtract:
    """Class to initialize DINO-like foundation model and extract features from an image.

    Note: implementation based on `DINOExtract` from https://github.com/google-research/omniglue/blob/main/src/omniglue/dino_extract.py.
    """

    def __init__(self, model: FoundationModelViT, feature_layer: int = 1, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.feature_layer = feature_layer
        self.image_size_max = 630

        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.forward(image)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Feeds image through DINO ViT model to extract features.

        :param image: (H, W, 3), decoded image bytes, value range [0, 255].
        :type image: np.ndarray
        :return: (H // 14, W // 14, C) image features.
        :rtype: np.ndarray
        """

        image = self._resize_input_image(image)
        image_processed = self._process_image(image)
        image_processed = image_processed.unsqueeze(0).float().to(self.device)
        features = self.extract_feature(image_processed)
        features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return features

    def _resize_input_image(self, image: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """Resizes image such that both dimensions are divisble by down_rate."""

        h_image, w_image = image.shape[:2]
        h_larger_flag = h_image > w_image
        large_side_image = max(h_image, w_image)

        # Resize the image with the largest side length smaller than a threshold
        # to accelerate ViT backbone inference (which has quadratic complexity).
        if large_side_image > self.image_size_max:
            if h_larger_flag:
                h_image_target = self.image_size_max
                w_image_target = int(self.image_size_max * w_image / h_image)
            else:
                w_image_target = self.image_size_max
                h_image_target = int(self.image_size_max * h_image / w_image)
        else:
            h_image_target = h_image
            w_image_target = w_image

        h, w = (
            h_image_target // self.h_down_rate,
            w_image_target // self.w_down_rate,
        )
        h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
        image = cv2.resize(image, (w_resize, h_resize), interpolation=interpolation)

        return image

    def _process_image(self, image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Turn image into pytorch tensor and normalize it using ImageNet mean/std."""

        mean = np.zeros((3,))
        std = np.ones((3,))
        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        image_processed = image / 255.0
        image_processed = (image_processed - mean) / std
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)

        return image_processed

    def extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts features from image.

        :param image: (B, 3, H, W) tensor.
        :type image: torch.Tensor
        :return: (B, C, H//14, W//14) image features.
        :rtype: torch.Tensor
        """

        b, _, h_origin, w_origin = image.shape
        out = self.model.get_intermediate_layers(image, n=self.feature_layer)[0]
        assert not isinstance(out, tuple)

        h = int(h_origin / self.h_down_rate)
        w = int(w_origin / self.w_down_rate)
        dim = out.shape[-1]
        out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()

        return out

    def _preprocess_shape(
        self,
        h_image: torch.Tensor,
        w_image: torch.Tensor,
        image_size_max: int = 630,
        h_down_rate: int = 14,
        w_down_rate: int = 14,
    ):
        # Flatten the tensors
        h_image = torch.squeeze(h_image)
        w_image = torch.squeeze(w_image)

        large_side_image = torch.maximum(h_image, w_image)

        h_image_target, w_image_target = h_image, w_image
        # Calculate new dimensions when height is larger
        if torch.greater(large_side_image, image_size_max) and torch.greater(h_image, w_image):
            h_image_target = torch.tensor(image_size_max, dtype=torch.int32).squeeze()
            w_image_target = (image_size_max * w_image / h_image).type(torch.int32)
        # Calculate new dimensions when width is larger or equal
        elif torch.greater(large_side_image, image_size_max):
            w_image_target = torch.tensor(image_size_max, dtype=torch.int32).squeeze()
            h_image_target = (image_size_max * h_image / w_image).type(torch.int32)

        # Resize to be divided by patch size
        h = h_image_target // h_down_rate
        w = w_image_target // w_down_rate
        h_resize = h * h_down_rate
        w_resize = w * w_down_rate

        # Expand dimensions
        assert isinstance(h_resize, torch.Tensor) and isinstance(w_resize, torch.Tensor)
        h_resize = torch.unsqueeze(h_resize, 0)
        w_resize = torch.unsqueeze(w_resize, 0)

        return h_resize, w_resize

    def _lookup_descriptor_bilinear(self, keypoint: np.ndarray, descriptor_map: np.ndarray) -> np.ndarray:
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
        self, dense_features: np.ndarray, kps: np.ndarray, image_dims: tuple[int, int], feat_dim: int = 768
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

        keypoints = torch.tensor(kps, dtype=torch.float32)
        height = torch.tensor(image_dims[0], dtype=torch.int32)
        width = torch.tensor(image_dims[1], dtype=torch.int32)
        feature_dim = torch.tensor(feat_dim, dtype=torch.int32)

        height_1d = torch.reshape(height, [1])
        width_1d = torch.reshape(width, [1])

        height_1d_resized, width_1d_resized = self._preprocess_shape(
            height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14
        )

        height_feat = height_1d_resized // 14
        width_feat = width_1d_resized // 14
        feature_dim_1d = torch.reshape(feature_dim, [1])

        size_feature = torch.concat([height_feat, width_feat, feature_dim_1d])
        dense_features_reshaped = torch.reshape(torch.tensor(dense_features), size_feature.tolist())

        img_size = torch.concat([width_1d, height_1d]).type(torch.float32)
        feature_size = torch.concat([width_feat, height_feat]).type(torch.float32)

        keypoints_feature = keypoints / torch.unsqueeze(img_size, dim=0) * torch.unsqueeze(feature_size, dim=0)

        dense_descriptors = []
        for kp in keypoints_feature:
            dense_descriptors.append(self._lookup_descriptor_bilinear(kp.numpy(), dense_features_reshaped.numpy()))
        dense_descriptors = np.array(dense_descriptors, dtype=np.float32)

        return dense_descriptors
