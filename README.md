# imguidedfilter-opencv

Implement imguidedfilter with OpenCV in Python

### why
the implementation in `cv2.ximgproc.guided_filter` gives different result compared to `imguidedfilter` in matlab, and given that I need to [recurr the paper](https://blog.0xbbc.com/2018/09/note-on-recurring-siggraph-2018-semantic-soft-segmentation), SIGGRAPH 2018 [Semantic Soft Segement](http://people.inf.ethz.ch/aksoyy/sss/) which uses `imguidedfilter`, I rewrite a `imguidedfilter` function with OpenCV with Python.

### usage

```python3
import cv2
import scipy.io as sio
import numpy as np
from imguidedfilter import imguidedfilter

image = cv2.imread('whatever.png')/255.0
features = sio.loadmat('features.mat')['features']

# Filter out super high numbers due to some instability in the network
features[features > 5] = 5
features[features < -5] = -5

# Filter each channel of features with image as the guide
fd = features.shape[2]
maxfd = fd - fd % 3
for i in range(0, maxfd, 3):
    # features(:, :, i : i+2) = imguidedfilter(features(:, :, i : i+2), image, 'NeighborhoodSize', 10);
    features[:, :, i : i + 3] = imguidedfilter(features[:, :, i : i + 3], image, (10, 10), 0.01)
for i in range(maxfd, fd):
    # features(:, :, i) = imguidedfilter(features(:, :, i), image, 'NeighborhoodSize', 10);
    features[:, :, i] = imguidedfilter(features[:, :, i], image, (10, 10), 0.01)

# Well, subsequently perhaps you may run PCA and normalize to [0, 1]
# simp = featurePCA(features, 3)
# for i in range(0, 3):
#     # simp(:,:,i) = simp(:,:,i) - min(min(simp(:,:,i)));
#     simp[:, :, i] = simp[:, :, i] - simp[:, :, i].min()
#     # simp(:,:,i) = simp(:,:,i) / max(max(simp(:,:,i)));
#     simp[:, :, i] = simp[:, :, i] / simp[:, :, i].max()
```

