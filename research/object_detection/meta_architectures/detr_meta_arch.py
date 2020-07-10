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

from object_detection.meta_architectures import detr_transformer
from object_detection.matchers import hungarian_matcher

class DETRMetaArch(model.DetectionModel):
    def __init__(self):
        self.num_queries = 100
        self.hidden_dimension = 100
        self.feature_extractor = faster_rcnn_resnet_keras_feature_extractor.FasterRCNNResnet50KerasFeatureExtractor(is_training=False)
        self.first_stage = self.feature_extractor.get_proposal_feature_extractor_model()
        self.target_assigner = target_assigner.create_target_assigner('DETR', 'detection')
        self.transformer = detr_transformer.Transformer()
        self.ffn = self.feature_extractor.get_box_classifier_feature_extractor_model()
        self.bboxes = tf.keras.layers.Dense(4)
        self.cls = tf.keras.layers.Dense(2)
        self.queries = tf.keras.Variable(tf.random([self.num_queries, self.hidden_dimension]))

    def predict(self, preprocessed_inputs, true_image_shapes, **side_inputs):
        x = self.first_stage(preprocessed_inputs)
        x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
        x = self.transformer([x, tf.repeat(tf.expand_dims(self.queries, 0), x.shape[0], axis=0)])
        x = self.ffn(x)
        return self.bboxes(x), self.cls(x)

    def loss(self, prediction_dict, true_image_shapes, scope=None):
        return 1

    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.

        See base class.

        For Faster R-CNN, we perform image resizing in the base class --- each
        class subclassing FasterRCNNMetaArch is responsible for any additional
        preprocessing (e.g., scaling pixel values to be in [-1, 1]).

        Args:
        inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.

        Returns:
        preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
        true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.
        Raises:
        ValueError: if inputs tensor does not have type tf.float32
        """

        with tf.name_scope('Preprocessor'):
            (resized_inputs,
            true_image_shapes) = shape_utils.resize_images_and_return_shapes(
                inputs, self._image_resizer_fn)

        return (self.feature_extractor.preprocess(resized_inputs),
                true_image_shapes)

    def restore_from_objects(self, fine_tune_checkpoint_type='detection'):
        raise NotImplementedError("Model restoration implemented yet.")

    def restore_map(self,
                    fine_tune_checkpoint_type='detection',
                    load_all_detection_checkpoint_vars=False):
        raise NotImplementedError("Model restoration implemented yet.")