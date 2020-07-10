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
from object_detection.core import losses

from object_detection.meta_architectures import detr_transformer
from object_detection.matchers import hungarian_matcher

class DETRMetaArch(model.DetectionModel):
  def __init__(self,
                is_training,
                num_classes,
                image_resizer_fn,
                feature_extractor,
                number_of_stages,
                first_stage_anchor_generator,
                first_stage_target_assigner,
                first_stage_atrous_rate,
                first_stage_box_predictor_arg_scope_fn,
                first_stage_box_predictor_kernel_size,
                first_stage_box_predictor_depth,
                first_stage_minibatch_size,
                first_stage_sampler,
                first_stage_non_max_suppression_fn,
                first_stage_max_proposals,
                first_stage_localization_loss_weight,
                first_stage_objectness_loss_weight,
                crop_and_resize_fn,
                initial_crop_size,
                maxpool_kernel_size,
                maxpool_stride,
                second_stage_target_assigner,
                second_stage_mask_rcnn_box_predictor,
                second_stage_batch_size,
                second_stage_sampler,
                second_stage_non_max_suppression_fn,
                second_stage_score_conversion_fn,
                second_stage_localization_loss_weight,
                second_stage_classification_loss_weight,
                second_stage_classification_loss,
                second_stage_mask_prediction_loss_weight=1.0,
                hard_example_miner=None,
                parallel_iterations=16,
                add_summaries=True,
                clip_anchors_to_image=False,
                use_static_shapes=False,
                resize_masks=True,
                freeze_batchnorm=False,
                return_raw_detections_during_predict=False,
                output_final_box_features=False):
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
        self._localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        self._classification_loss = losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)
        self._second_stage_loc_loss_weight = second_stage_localization_loss_weight
        self._second_stage_cls_loss_weight = second_stage_classification_loss_weight

  def predict(self, preprocessed_inputs, true_image_shapes, **side_inputs):
        x = self.first_stage(preprocessed_inputs)
        x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
        x = self.transformer([x, tf.repeat(tf.expand_dims(self.queries, 0), x.shape[0], axis=0)])
        x = self.ffn(x)
        return self.bboxes(x), self.cls(x)

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

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Compute scalar loss tensors given prediction tensors.

    If number_of_stages=1, only RPN related losses are computed (i.e.,
    `rpn_localization_loss` and `rpn_objectness_loss`).  Otherwise all
    losses are computed.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        `image_shape`, and `anchors` fields.  Otherwise we expect
        prediction_dict to additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`, and
        `proposal_boxes` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`, 'second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      (groundtruth_boxlists, groundtruth_classes_with_background_list,
       groundtruth_masks_list, groundtruth_weights_list
      ) = self._format_groundtruth_data(
          self._image_batch_shape_2d(prediction_dict['image_shape']))
      loss_dict = self._loss_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'], groundtruth_boxlists,
            groundtruth_classes_with_background_list,
            groundtruth_weights_list, prediction_dict['image_shape'],
            prediction_dict.get('mask_predictions'), groundtruth_masks_list,
            prediction_dict.get(
                fields.DetectionResultFields.detection_boxes),
            prediction_dict.get(
                fields.DetectionResultFields.num_detections))
    return loss_dict

  def _loss_box_classifier(self,
                           refined_box_encodings,
                           class_predictions_with_background,
                           proposal_boxes,
                           num_proposals,
                           groundtruth_boxlists,
                           groundtruth_classes_with_background_list,
                           groundtruth_weights_list,
                           image_shape,
                           prediction_masks=None,
                           groundtruth_masks_list=None,
                           detection_boxes=None,
                           num_detections=None):
    """Computes scalar box classifier loss tensors.

    Uses self._detector_target_assigner to obtain regression and classification
    targets for the second stage box classifier, optionally performs
    hard mining, and returns losses.  All losses are computed independently
    for each image and then averaged across the batch.
    Please note that for boxes and masks with multiple labels, the box
    regression and mask prediction losses are only computed for one label.

    This function assumes that the proposal boxes in the "padded" regions are
    actually zero (and thus should not be matched to).


    Args:
      refined_box_encodings: a 3-D tensor with shape
        [total_num_proposals, num_classes, box_coder.code_size] representing
        predicted (final) refined box encodings. If using a shared box across
        classes this will instead have shape
        [total_num_proposals, 1, box_coder.code_size].
      class_predictions_with_background: a 2-D tensor with shape
        [total_num_proposals, num_classes + 1] containing class
        predictions (logits) for each of the anchors.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: [batch_size, self.num_queries, 4] representing
        decoded proposal bounding boxes.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      groundtruth_boxlists: a list of BoxLists containing coordinates of the
        groundtruth boxes.
      groundtruth_classes_with_background_list: a list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes + 1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      image_shape: a 1-D tensor of shape [4] representing the image shape.
      prediction_masks: an optional 4-D tensor with shape [total_num_proposals,
        num_classes, mask_height, mask_width] containing the instance masks for
        each box.
      groundtruth_masks_list: an optional list of 3-D tensors of shape
        [num_boxes, image_height, image_width] containing the instance masks for
        each of the boxes.
      detection_boxes: 3-D float tensor of shape [batch,
        max_total_detections, 4] containing post-processed detection boxes in
        normalized co-ordinates.
      num_detections: 1-D int32 tensor of shape [batch] containing number of
        valid detections in `detection_boxes`.

    Returns:
      a dictionary mapping loss keys ('second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.

    Raises:
      ValueError: if `predict_instance_masks` in
        second_stage_mask_rcnn_box_predictor is True and
        `groundtruth_masks_list` is not provided.
    """
    with tf.name_scope('BoxClassifierLoss'):
      paddings_indicator = self._padded_batched_proposals_indicator(
          num_proposals, proposal_boxes.shape[1])
      proposal_boxlists = [
          box_list.BoxList(proposal_boxes_single_image)
          for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
      batch_size = len(proposal_boxlists)

      num_proposals_or_one = tf.cast(tf.expand_dims(
          tf.maximum(num_proposals, tf.ones_like(num_proposals)), 1),
                                     dtype=tf.float32)
      normalizer = tf.tile(num_proposals_or_one,
                           [1, self.num_queries]) * batch_size

      (batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, _) = target_assigner.batch_assign_targets(
           target_assigner=self.target_assigner,
           anchors_batch=proposal_boxlists,
           gt_box_batch=groundtruth_boxlists,
           gt_class_targets_batch=groundtruth_classes_with_background_list,
           unmatched_class_label=tf.constant(
               [1] + self._num_classes * [0], dtype=tf.float32),
           gt_weights_batch=groundtruth_weights_list)

      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [batch_size, self.num_queries, -1])

      flat_cls_targets_with_background = tf.reshape(
          batch_cls_targets_with_background,
          [batch_size * self.num_queries, -1])
      one_hot_flat_cls_targets_with_background = tf.argmax(
          flat_cls_targets_with_background, axis=1)
      one_hot_flat_cls_targets_with_background = tf.one_hot(
          one_hot_flat_cls_targets_with_background,
          flat_cls_targets_with_background.get_shape()[1])

      # If using a shared box across classes use directly
      if refined_box_encodings.shape[1] == 1:
        reshaped_refined_box_encodings = tf.reshape(
            refined_box_encodings,
            [batch_size, self.num_queries, self._box_coder.code_size])
      # For anchors with multiple labels, picks refined_location_encodings
      # for just one class to avoid over-counting for regression loss and
      # (optionally) mask loss.
      else:
        reshaped_refined_box_encodings = (
            self._get_refined_encodings_for_postitive_class(
                refined_box_encodings,
                one_hot_flat_cls_targets_with_background, batch_size))

      losses_mask = None
      if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
        losses_mask = tf.stack(self.groundtruth_lists(
            fields.InputDataFields.is_annotated))
      second_stage_loc_losses = self._localization_loss(
          reshaped_refined_box_encodings,
          batch_reg_targets,
          weights=batch_reg_weights,
          losses_mask=losses_mask) / normalizer
      second_stage_cls_losses = ops.reduce_sum_trailing_dimensions(
          self._classification_loss(
              class_predictions_with_background,
              batch_cls_targets_with_background,
              weights=batch_cls_weights,
              losses_mask=losses_mask),
          ndims=2) / normalizer

      second_stage_loc_loss = tf.reduce_sum(
          second_stage_loc_losses * tf.cast(paddings_indicator,
                                            dtype=tf.float32))
      second_stage_cls_loss = tf.reduce_sum(
          second_stage_cls_losses * tf.cast(paddings_indicator,
                                            dtype=tf.float32))

      localization_loss = tf.multiply(self._second_stage_loc_loss_weight,
                                      second_stage_loc_loss,
                                      name='localization_loss')

      classification_loss = tf.multiply(self._second_stage_cls_loss_weight,
                                        second_stage_cls_loss,
                                        name='classification_loss')

      loss_dict = {'Loss/BoxClassifierLoss/localization_loss':
                       localization_loss,
                   'Loss/BoxClassifierLoss/classification_loss':
                       classification_loss}
      second_stage_mask_loss = None
    return loss_dict


################################## UTILITY FUNCTIONS ########################################

def _format_groundtruth_data(self, image_shapes):
    """Helper function for preparing groundtruth data for target assignment.

    In order to be consistent with the model.DetectionModel interface,
    groundtruth boxes are specified in normalized coordinates and classes are
    specified as label indices with no assumed background category.  To prepare
    for target assignment, we:
    1) convert boxes to absolute coordinates,
    2) add a background class at class index 0
    3) groundtruth instance masks, if available, are resized to match
       image_shape.

    Args:
      image_shapes: a 2-D int32 tensor of shape [batch_size, 3] containing
        shapes of input image in the batch.

    Returns:
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_masks_list: If present, a list of 3-D tf.float32 tensors of
        shape [num_boxes, image_height, image_width] containing instance masks.
        This is set to None if no masks exist in the provided groundtruth.
    """
    # pylint: disable=g-complex-comprehension
    groundtruth_boxlists = [
        box_list_ops.to_absolute_coordinates(
            box_list.BoxList(boxes), image_shapes[i, 0], image_shapes[i, 1])
        for i, boxes in enumerate(
            self.groundtruth_lists(fields.BoxListFields.boxes))
    ]
    groundtruth_classes_with_background_list = []
    for one_hot_encoding in self.groundtruth_lists(
        fields.BoxListFields.classes):
      groundtruth_classes_with_background_list.append(
          tf.cast(
              tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'),
              dtype=tf.float32))

    groundtruth_masks_list = self._groundtruth_lists.get(
        fields.BoxListFields.masks)
    # TODO(rathodv): Remove mask resizing once the legacy pipeline is deleted.
    if groundtruth_masks_list is not None and self._resize_masks:
      resized_masks_list = []
      for mask in groundtruth_masks_list:

        _, resized_mask, _ = self._image_resizer_fn(
            # Reuse the given `image_resizer_fn` to resize groundtruth masks.
            # `mask` tensor for an image is of the shape [num_masks,
            # image_height, image_width]. Below we create a dummy image of the
            # the shape [image_height, image_width, 1] to use with
            # `image_resizer_fn`.
            image=tf.zeros(tf.stack([tf.shape(mask)[1],
                                     tf.shape(mask)[2], 1])),
            masks=mask)
        resized_masks_list.append(resized_mask)

      groundtruth_masks_list = resized_masks_list
    # Masks could be set to bfloat16 in the input pipeline for performance
    # reasons. Convert masks back to floating point space here since the rest of
    # this module assumes groundtruth to be of float32 type.
    float_groundtruth_masks_list = []
    if groundtruth_masks_list:
      for mask in groundtruth_masks_list:
        float_groundtruth_masks_list.append(tf.cast(mask, tf.float32))
      groundtruth_masks_list = float_groundtruth_masks_list

    if self.groundtruth_has_field(fields.BoxListFields.weights):
      groundtruth_weights_list = self.groundtruth_lists(
          fields.BoxListFields.weights)
    else:
      # Set weights for all batch elements equally to 1.0
      groundtruth_weights_list = []
      for groundtruth_classes in groundtruth_classes_with_background_list:
        num_gt = tf.shape(groundtruth_classes)[0]
        groundtruth_weights = tf.ones(num_gt)
        groundtruth_weights_list.append(groundtruth_weights)

    return (groundtruth_boxlists, groundtruth_classes_with_background_list,
            groundtruth_masks_list, groundtruth_weights_list)

  def _image_batch_shape_2d(self, image_batch_shape_1d):
    """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.

    Example:
    If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
    image batch tensor would be [[300, 300, 3], [300, 300, 3]]

    Args:
      image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].

    Returns:
      image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row is
        of the form [height, width, channels].
    """
    return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                   [image_batch_shape_1d[0], 1])

  def _padded_batched_proposals_indicator(self,
                                          num_proposals,
                                          max_num_proposals):
    """Creates indicator matrix of non-pad elements of padded batch proposals.

    Args:
      num_proposals: Tensor of type tf.int32 with shape [batch_size].
      max_num_proposals: Maximum number of proposals per image (integer).

    Returns:
      A Tensor of type tf.bool with shape [batch_size, max_num_proposals].
    """
    batch_size = tf.size(num_proposals)
    tiled_num_proposals = tf.tile(
        tf.expand_dims(num_proposals, 1), [1, max_num_proposals])
    tiled_proposal_index = tf.tile(
        tf.expand_dims(tf.range(max_num_proposals), 0), [batch_size, 1])
    return tf.greater(tiled_num_proposals, tiled_proposal_index)

  def _get_refined_encodings_for_postitive_class(
      self, refined_box_encodings, flat_cls_targets_with_background,
      batch_size):
    # We only predict refined location encodings for the non background
    # classes, but we now pad it to make it compatible with the class
    # predictions
    refined_box_encodings_with_background = tf.pad(refined_box_encodings,
                                                   [[0, 0], [1, 0], [0, 0]])
    refined_box_encodings_masked_by_class_targets = (
        box_list_ops.boolean_mask(
            box_list.BoxList(
                tf.reshape(refined_box_encodings_with_background,
                           [-1, self._box_coder.code_size])),
            tf.reshape(tf.greater(flat_cls_targets_with_background, 0), [-1]),
            use_static_shapes=self._use_static_shapes,
            indicator_sum=batch_size * self.num_queries
            if self._use_static_shapes else None).get())
    return tf.reshape(
        refined_box_encodings_masked_by_class_targets, [
            batch_size, self.num_queries,
            self._box_coder.code_size
        ])