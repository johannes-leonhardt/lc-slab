# LC-SLab - An Object-based Deep Learning Framework for Large-scale Land Cover Classification from Satellite Imagery and Sparse In-situ Labels

This repository contains the code for our paper "LC-SLab - An Object-based Deep Learning Framework for Large-scale Land Cover Classification from Satellite Imagery and Sparse In-situ Labels", which is currently under review.

## Abstract

Large-scale land cover maps generated using deep learning play a critical role in data-driven analysis and decision-making across a wide range of Earth science applications. Open in-situ datasets from principled land cover surveys offer a scalable alternative to manual annotation for training such models. However, their sparse spatial coverage leads to fragmented and noisy predictions when used with existing deep learning-based land cover mapping approaches. A promising direction to address this issue is object-based classification, which assigns labels to semantically coherent image regions rather than individual pixels, thereby imposing a minimum mapping unit that controls spatial fragmentation. Despite this potential, object-based methods remain underexplored in deep learning-based land cover mapping pipelines, especially in the context of medium-resolution imagery and sparse supervision.

To address this gap, we propose LC-SLab, the first deep learning framework for systematically exploring object-based deep learning methods for large-scale land cover classification under sparse supervision. LC-SLab supports both input-level aggregation via graph neural networks, and output-level aggregation by postprocessing results from established semantic segmentation models. Additionally, we incorporate features from a large pre-trained network to improve performance on small datasets.

We evaluate the framework on annual Sentinel-2 composites with sparse LUCAS labels, focusing on the tradeoff between accuracy and fragmentation, as well as sensitivity to dataset size. Our results show that object-based methods can match or exceed the accuracy of common pixel-wise models while producing substantially more coherent maps. Input-level aggregation proves more robust on smaller datasets, whereas output-level aggregation performs best with more data. Several configurations of LC-SLab also outperform existing land cover products, highlighting the framework's practical utility.

## Instructions

- The dataset used in the study can be downloaded by running the `download_data.sh` script.
- Install the dependencies, as provided in the `requirements.txt` file.

Afterwards, you can:
- Train a new classifier on LUCAS data with `train_classifier.py`.
- Test your classifier with `test_classifier.py`.
- Train a classifier on ESA WorldCover, as used for the feature extractor with `pretrain_classifier.py`.
- Evaluate the agreement between third-party products and LUCAS with `test_third_party_products.py`.

