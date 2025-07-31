# Omniglue

This repository provides a more flexible version of [Omniglue](https://doi.org/10.1109/CVPR52733.2024.01878), based on the [Google's original implementation](https://github.com/google-research/omniglue).

The main difference from Google's version is that the matching model has been decoupled from the sparse and dense feature extractors. Thanks to this modular design, you can use any type of sparse or dense feature extractor, as long as their outputs match the expected inputs of `FlexibleOmniGlue`.

To further support this flexible architecture, a helper class named `DenseExtract` is provided. It facilitates the computation and transformation of dense features to match the required input format.

## Installation

Install this repository using `pip`:

```bash
git clone https://github.com/aelskens/omniglue.git && cd omniglue
pip install .
```

To download the OmniGlue's weights, use the following commands (refer to [original OmniGlue repository](https://github.com/google-research/omniglue) for more details):

```bash
wget https://storage.googleapis.com/omniglue/og_export.zip
unzip og_export.zip && rm og_export.zip
```

## Usage

Below is a minimal example using OpenCV SIFT as a sparse feature extractor and DINOv2 as a dense feature extractor. Note that since **OmniGlue expects a 256-dimensional descriptor** for each keypoint, the 128-dimensional SIFT descriptors were simply duplicated to meet this requirement.

```python
import cv2
import numpy as np
import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imread

from omniglue import DenseExtract, FlexibleOmniGlue

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load all models
sparse_extractor = cv2.SIFT.create()
dense_extractor = DenseExtract(
    model=torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14"),
)
matcher = FlexibleOmniGlue()
matcher.load("./models/og_export")

# Load the images
src = imread("/data/dataset_ANHIR/images/breast_3/scale-2.5pc/HE.png")
dst = imread("/data/dataset_ANHIR/images/breast_3/scale-2.5pc/HER2.png")

# Compute the sparse and dense features
src_sparse_kp, src_sparse_desc = sparse_extractor.compute(
    img_as_ubyte(rgb2gray(src)), keypoints=sparse_extractor.detect(img_as_ubyte(rgb2gray(src)))
)
src_sparse_kp, src_sparse_kp_score = np.array([kp.pt for kp in src_sparse_kp]), np.array(
    [kp.response for kp in src_sparse_kp]
)
dst_sparse_kp, dst_sparse_desc = sparse_extractor.compute(
    img_as_ubyte(rgb2gray(dst)), keypoints=sparse_extractor.detect(img_as_ubyte(rgb2gray(dst)))
)
dst_sparse_kp, dst_sparse_kp_score = np.array([kp.pt for kp in dst_sparse_kp]), np.array(
    [kp.response for kp in dst_sparse_kp]
)
src_dense_map = dense_extractor(img_as_ubyte(src))
dst_dense_map = dense_extractor(img_as_ubyte(dst))
src_dense_kp = dense_extractor.get_dense_descriptors_from_sparse_keypoints(
    src_dense_map,
    src_sparse_kp,
    src.shape[:2],
    768,
)
dst_dense_kp = dense_extractor.get_dense_descriptors_from_sparse_keypoints(
    dst_dense_map,
    dst_sparse_kp,
    dst.shape[:2],
    768,
)

# Match the features
indices, _ = matcher(
    img0_shape_rc=src.shape[:2],
    img1_shape_rc=dst.shape[:2],
    img0_sparse_keypoints=src_sparse_kp,
    img0_sparse_descriptors=np.hstack([src_sparse_desc, src_sparse_desc]),
    img1_sparse_keypoints=dst_sparse_kp,
    img1_sparse_descriptors=np.hstack([dst_sparse_desc, dst_sparse_desc]),
    img0_dense_descriptors=src_dense_kp.numpy(),
    img1_dense_descriptors=dst_dense_kp.numpy(),
    img0_sparse_score=src_sparse_kp_score,
    img1_sparse_score=dst_sparse_kp_score,
)
src_matched_kp = src_sparse_kp[indices[:, 0], :]
dst_matched_kp = dst_sparse_kp[indices[:, 1], :]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(src)
plt.plot(src_matched_kp[:, 0], src_matched_kp[:, 1], "r.")
plt.subplot(1, 2, 2)
plt.imshow(dst)
plt.plot(dst_matched_kp[:, 0], dst_matched_kp[:, 1], "r.")
plt.show()
```

Make sure to provide the correct path to the `og_export` folder when calling the matcher's `load` method, and update the paths to the input images accordingly.