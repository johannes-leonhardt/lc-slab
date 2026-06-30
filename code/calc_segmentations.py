import os

from data.segmentation import Segmentor

data_root = os.path.join(os.path.sep, "data2", "jleonhardt", "LC-SLab")

# SLIC segmentations
segmentation_method = "SLIC"
for n_segments in [819, 409, 204, 102]:
    segmentation_kwargs = {"n_segments": n_segments, "compactness": 0.1, "convert2lab": False, "min_size_factor": 1, "start_label":0}

# Felzenszwalb segmentations
# segmentation_method = "Felzenszwalb"
# for min_size in [5, 10, 20, 40]:
#     segmentation_kwargs = {"scale": 1, "sigma": 0, "min_size": min_size}

    segmenter = Segmentor(
        segmentation_method = segmentation_method,
        segmentation_kwargs = segmentation_kwargs,
        normalization_method = "Gamma",
        normalization_kwargs = {"gamma": 0.5}
    )

    segmenter.segment_all_and_save(data_root)