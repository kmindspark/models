import abc
import collections
import functools
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import shape_utils
from object_detection.models import faster_rcnn_resnet_keras_feature_extractor

from object_detection.meta_architectures import transformer
from object_detection.matchers import hungarian_matcher

class DETRMetaArch(model.DetectionModel):
    def __init__(self):
        self.feature_extractor = faster_rcnn_resnet_keras_feature_extractor.FasterRCNNResnet50KerasFeatureExtractor(is_training=False)
        self.first_stage = self.feature_extractor.get_proposal_feature_extractor_model()
        self.target_assigner = target_assigner.create_target_assigner('DETR', 'detection')
        self.transformer = transformer.Transformer()
        self.ffn = self.feature_extractor.get_box_classifier_feature_extractor_model()
        
    def __call__(self, inputs):
        inputs = self.first_stage(inputs)
        print(inputs)