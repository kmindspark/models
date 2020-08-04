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
from object_detection.core import box_predictor
from object_detection.utils import shape_utils
from object_detection.utils import ops
from object_detection.models import faster_rcnn_resnet_keras_feature_extractor
from object_detection.core import losses
from object_detection.utils import variables_helper

#from object_detection.meta_architectures import detr_lib
#from object_detection.meta_architectures import detr_lib_revert
from object_detection.meta_architectures import detr_transformer
from object_detection.matchers import hungarian_matcher
from object_detection.core import post_processing

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
    print("Initializing model...")
    super(DETRMetaArch, self).__init__(num_classes=num_classes)
    self._image_resizer_fn = image_resizer_fn
    self.num_queries = 100
    self.hidden_dimension = 256
    self.feature_extractor = faster_rcnn_resnet_keras_feature_extractor.FasterRCNNResnet50KerasFeatureExtractor(is_training=is_training)#, weight_decay=0.0001)
    self.first_stage = self.feature_extractor.get_proposal_feature_extractor_model()
    #for layer in self.first_stage.layers:
    #  layer.trainable = False
    self.target_assigner = target_assigner.create_target_assigner('DETR', 'detection')
    #self.transformer = #detr_lib.Transformer(attention_dropout=0.0, layer_postprocess_dropout=0.0,
    #relu_dropout=0.0, hidden_size=self.hidden_dimension, filter_size=self.hidden_dimension,
    #num_hidden_layers=3)#self.transformer_args) #hidden_size=self.hidden_dimension, filter_size=self.hidden_dimension)
    self.transformer_args = {"hidden_size": self.hidden_dimension, "attention_dropout": 0, "num_heads": 8, "layer_postprocess_dropout": 0, "dtype": tf.float32, 
      "num_hidden_layers": 6, "filter_size": 256, "relu_dropout": 0}
    self.transformer = detr_transformer.Transformer(self.transformer_args)#detr_lib.Transformer(attention_dropout=0.0, layer_postprocess_dropout=0.0, relu_dropout=0.0)
    #self.ffn = self.feature_extractor.get_box_classifier_feature_extractor_model()
    #self.bboxes = tf.keras.layers.Dense(4)
    self.cls = tf.keras.layers.Dense(num_classes + 1)
    self.cls_activation = tf.keras.layers.Softmax()
    print("INITIALIZING QUERIES")
    self.queries = tf.keras.backend.variable(tf.random.uniform([self.num_queries, self.hidden_dimension]))  #tf.keras.backend.variable(value=tf.random_normal_initializer(stddev=1.0)([self.num_queries, self.hidden_dimension]), name="object_queries", dtype=tf.float32)# tf.random_normal_initializer tf.keras.backend.variable(tf.zeros([self.num_queries, self.hidden_dimension]), name="object_queries") #tf.zeros([self.num_queries, self.hidden_dimension]), dtype=tf.float32) #tf.random.uniform([self.num_queries, self.hidden_dimension]) tf.Variable(initial_value=tf.zeros((self.num_queries, self.hidden_dimension)), trainable=True)
    print(self.queries)
    self._localization_loss = losses.WeightedSmoothL1LocalizationLoss()
    self._localization_loss_iou = losses.WeightedGIOULocalizationLoss()
    self._classification_loss = losses.WeightedSoftmaxClassificationLoss()
    self._second_stage_loc_loss_weight = second_stage_localization_loss_weight
    self._second_stage_cls_loss_weight = second_stage_classification_loss_weight
    self._box_coder = self.target_assigner.get_box_coder()
    self._parallel_iterations = parallel_iterations
    self._post_filter = tf.keras.layers.Conv2D(self.hidden_dimension, 1)
    self._second_stage_nms_fn = second_stage_non_max_suppression_fn
    self._box_ffn = tf.keras.Sequential(layers=[tf.keras.layers.Dense(self.hidden_dimension, activation="relu"),
                                                tf.keras.layers.Dense(4, activation="sigmoid")])
    self.is_training = is_training
    self._second_stage_score_conversion_fn = second_stage_score_conversion_fn
    print("CONSTRUCTOR TRAINING", self.is_training)
  @property
  def first_stage_feature_extractor_scope(self):
    return 'FirstStageFeatureExtractor'

  @property
  def second_stage_feature_extractor_scope(self):
    return 'SecondStageFeatureExtractor'


  def predict(self, preprocessed_inputs, true_image_shapes, **side_inputs):
    #if not self.is_training:
    #  self.queries = tf.keras.backend.variable(value=tf.random_normal_initializer(stddev=20.0)([self.num_queries, self.hidden_dimension]), name="object_queries", dtype=tf.float32)

    image_shape = tf.shape(preprocessed_inputs)
    with tf.name_scope("FirstStage"):
      x = self.first_stage(preprocessed_inputs, training=self.is_training)
    x = self._post_filter(x)
    x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
    x = self.transformer([x, tf.repeat(tf.expand_dims(self.queries, 0), x.shape[0], axis=0)], training=self.is_training)
    bboxes_encoded, logits = self._box_ffn(x), self.cls(x)

    print("Actual bboxes", bboxes_encoded)
    print("Actually predicted logits: ", logits)
    print("Queries", self.queries)

    #if (not self.is_training):
    #  fake_logits = np.zeros((x.shape[0], 100, self.num_classes + 1))
    #  fake_logits[:,:,1] = 1000
    #  logits = tf.convert_to_tensor(fake_logits, dtype=tf.float32)

    reshaped_bboxes = tf.reshape(bboxes_encoded, [bboxes_encoded.shape[0] * bboxes_encoded.shape[1], 1, bboxes_encoded.shape[2]])
    batches_queries = tf.repeat(tf.expand_dims(self.num_queries, 0), x.shape[0], axis=0)
    return {
      "refined_box_encodings": reshaped_bboxes,
      "class_predictions_with_background": logits,
      "num_proposals": batches_queries,
      "proposal_boxes": bboxes_encoded,
      "image_shape": image_shape
    }

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
    """Returns a map of Trackable objects to load from a foreign checkpoint.

    Returns a dictionary of Tensorflow 2 Trackable objects (e.g. tf.Module
    or Checkpoint). This enables the model to initialize based on weights from
    another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Note that this function is intended to be used to restore Keras-based
    models when running Tensorflow 2, whereas restore_map (above) is intended
    to be used to restore Slim-based models when running Tensorflow 1.x.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.

    Returns:
      A dict mapping keys to Trackable objects (tf.Module or Checkpoint).
    """
    if fine_tune_checkpoint_type == 'classification':
      return {
          'feature_extractor':
              self.feature_extractor.classification_backbone
      }

  def restore_map(self,
                  fine_tune_checkpoint_type='classification',
                  load_all_detection_checkpoint_vars=False):
    print("NOT FOR TF 2")
    
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
           gt_weights_batch=groundtruth_weights_list,
           class_predictions=class_predictions_with_background)

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

      print("LOSS: encodings and targets")
      print(reshaped_refined_box_encodings)
      print(batch_reg_targets)
      print(class_predictions_with_background)
      print(batch_cls_targets_with_background)
      second_stage_loc_losses = 5 * self._localization_loss(
          reshaped_refined_box_encodings,
          batch_reg_targets,
          weights=batch_reg_weights,
          losses_mask=losses_mask) / normalizer

      def convert_to_minmaxcoords(input_tensor):
        reshaped_encodings = tf.reshape(input_tensor, [-1, 4])
        ycenter = tf.gather(reshaped_encodings, [0], axis=1)
        xcenter = tf.gather(reshaped_encodings, [1], axis=1)
        h = tf.gather(reshaped_encodings, [2], axis=1)
        w = tf.gather(reshaped_encodings, [3], axis=1)
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        #print("RESULT", tf.stack([ymin, xmin, ymax, xmax], axis=1))
        return tf.squeeze(tf.stack([ymin, xmin, ymax, xmax], axis=1))

      my_loc_loss = self._localization_loss_iou(
          convert_to_minmaxcoords(tf.reshape(reshaped_refined_box_encodings, [-1, 4])),
          convert_to_minmaxcoords(tf.reshape(batch_reg_targets, [-1, 4])),
          weights=batch_reg_weights,
          losses_mask=losses_mask)
      my_loc_loss = tf.reshape(my_loc_loss, shape=[reshaped_refined_box_encodings.shape[0], reshaped_refined_box_encodings.shape[1]])
      second_stage_loc_losses += 2 * my_loc_loss/normalizer

      batch_cls_weights = tf.concat([tf.expand_dims(batch_cls_weights[:, :, 0] / 10, axis=2), batch_cls_weights[:, :, 1:]], axis=-1)

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
    return loss_dict

  def updates(self):
    """Returns a list of update operators for this model.

    Returns a list of update operators for this model that must be executed at
    each training step. The estimator's train op needs to have a control
    dependency on these updates.

    Returns:
      A list of update operators.
    """
    raise NotImplementedError("This function should only be called in TF 1.x")

  def regularization_losses(self):
    return []
    #all_losses = []
    #if self.first_stage:
    #  all_losses.extend(self.first_stage.losses)

  def postprocess(self, prediction_dict, true_image_shapes):
    """Convert prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results.
    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_converter is used,
    then scores are remapped (and may thus have a different interpretation).

    If number_of_stages=1, the returned results represent proposals from the
    first stage RPN and are padded to have self.num_queries for each
    image; otherwise, the results can be interpreted as multiclass detections
    from the full two-stage model and are padded to self._max_detections.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        and `anchors` fields.  Otherwise we expect prediction_dict to
        additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`,
        `proposal_boxes` and, optionally, `mask_predictions` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detection, 4]
        detection_scores: [batch, max_detections]
        detection_multiclass_scores: [batch, max_detections, 2]
        detection_anchor_indices: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (this entry is only created if rpn_mode=False)
        num_detections: [batch]
        raw_detection_boxes: [batch, total_detections, 4]
        raw_detection_scores: [batch, total_detections, num_classes + 1]

    Raises:
      ValueError: If `predict` is called before `preprocess`.
      ValueError: If `_output_final_box_features` is true but
        rpn_features_to_crop is not in the prediction_dict.
    """
    with tf.name_scope('SecondStagePostprocessor'):
      detections_dict = self._postprocess_box_classifier_new(
          prediction_dict['refined_box_encodings'],
          prediction_dict['class_predictions_with_background'],
          prediction_dict['proposal_boxes'],
          prediction_dict['num_proposals'],
          true_image_shapes,
          orig_image_shapes=prediction_dict['image_shape'])

    return detections_dict

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
        #box_list_ops.to_absolute_coordinates(
        #    box_list.BoxList(boxes), image_shapes[i, 0], image_shapes[i, 1])
        box_list.BoxList(boxes)
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
            None, groundtruth_weights_list)

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

  def _postprocess_box_classifier(self,
                                  refined_box_encodings,
                                  class_predictions_with_background,
                                  proposal_boxes,
                                  num_proposals,
                                  image_shapes,
                                  mask_predictions=None,
                                  orig_image_shapes=None):
    """Converts predictions from the second stage box classifier to detections.

    Args:
      refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
      class_predictions_with_background: a 2-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.num_queries, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
      num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
      image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
      mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4] in normalized co-ordinates.
        `detection_scores`: [batch, max_detections]
        `detection_multiclass_scores`: [batch, max_detections,
          num_classes_with_background] tensor with class score distribution for
          post-processed detection boxes including background class if any.
        `detection_anchor_indices`: [batch, max_detections] with anchor
          indices.
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
          (optional) [batch, max_detections, mask_height, mask_width]. Note
          that a pixel-wise sigmoid score converter is applied to the detection
          masks.
        `raw_detection_boxes`: [batch, total_detections, 4] tensor with decoded
          detection boxes in normalized coordinates, before Non-Max Suppression.
          The value total_detections is the number of second stage anchors
          (i.e. the total number of boxes before NMS).
        `raw_detection_scores`: [batch, total_detections,
          num_classes_with_background] tensor of multi-class scores for
          raw detection boxes. The value total_detections is the number of
          second stage anchors (i.e. the total number of boxes before NMS).
    """
    #print("ORIG: ", refined_box_encodings)
    refined_box_encodings_batch = tf.reshape(
        refined_box_encodings,
        [-1,
        self.num_queries,
        refined_box_encodings.shape[1],
        self._box_coder.code_size])
    print(refined_box_encodings_batch)
    class_predictions_with_background_batch = tf.reshape(
        class_predictions_with_background,
        [-1, self.num_queries, self.num_classes + 1]
    )
    refined_decoded_boxes_batch = self._batch_decode_boxes(
        refined_box_encodings_batch, proposal_boxes)
    print(refined_decoded_boxes_batch)
    refined_decoded_boxes_batch = ops.normalized_to_image_coordinates(tf.squeeze(refined_decoded_boxes_batch, axis=[2]), image_shape=orig_image_shapes, temp=True)
    refined_decoded_boxes_batch = tf.expand_dims(refined_decoded_boxes_batch, axis=2)
    class_predictions_with_background_batch_normalized = self._second_stage_score_conversion_fn(
            class_predictions_with_background_batch)#class_predictions_with_background_batch #(
        #self._second_stage_score_conversion_fn(
        #    class_predictions_with_background_batch))
    class_predictions_batch = tf.reshape(
        tf.slice(class_predictions_with_background_batch_normalized,
                [0, 0, 1], [-1, -1, -1]),
        [-1, self.num_queries, self.num_classes])
    clip_window = self._compute_clip_window(image_shapes)
    mask_predictions_batch = None

    batch_size = shape_utils.combined_static_and_dynamic_shape(
        refined_box_encodings_batch)[0]
    batch_anchor_indices = tf.tile(
        tf.expand_dims(tf.range(self.num_queries), 0),
        multiples=[batch_size, 1])
    additional_fields = {
        'multiclass_scores': class_predictions_with_background_batch_normalized,
        'anchor_indices': tf.cast(batch_anchor_indices, tf.float32)
    }
    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
    nmsed_additional_fields, num_detections) = self._second_stage_nms_fn(
        refined_decoded_boxes_batch,
        class_predictions_batch,
        clip_window=clip_window,
        change_coordinate_frame=True,
        num_valid_boxes=num_proposals,
        additional_fields=additional_fields,
        masks=mask_predictions_batch)
    print("BEFORE", refined_decoded_boxes_batch)
    print("AFTER", nmsed_boxes)
    if refined_decoded_boxes_batch.shape[2] > 1:
      class_ids = tf.expand_dims(
          tf.argmax(class_predictions_with_background_batch[:, :, 1:], axis=2,
                    output_type=tf.int32),
          axis=-1)
      raw_detection_boxes = tf.squeeze(
          tf.batch_gather(refined_decoded_boxes_batch, class_ids), axis=2)
    else:
      raw_detection_boxes = tf.squeeze(refined_decoded_boxes_batch, axis=2)

    raw_normalized_detection_boxes = shape_utils.static_or_dynamic_map_fn(
        self._normalize_and_clip_boxes,
        elems=[raw_detection_boxes, image_shapes],
        dtype=tf.float32)

    detections = {
        fields.DetectionResultFields.detection_boxes:
            nmsed_boxes,
        fields.DetectionResultFields.detection_scores:
            nmsed_scores,
        fields.DetectionResultFields.detection_classes:
            nmsed_classes,
        fields.DetectionResultFields.detection_multiclass_scores:
            nmsed_additional_fields['multiclass_scores'],
        fields.DetectionResultFields.detection_anchor_indices:
            tf.cast(nmsed_additional_fields['anchor_indices'], tf.int32),
        fields.DetectionResultFields.num_detections:
            tf.cast(num_detections, dtype=tf.float32),
        fields.DetectionResultFields.raw_detection_boxes:
            raw_normalized_detection_boxes,
        fields.DetectionResultFields.raw_detection_scores:
            class_predictions_with_background_batch_normalized
    }
    return detections

  def _postprocess_box_classifier_new(self,
                                  refined_box_encodings,
                                  class_predictions_with_background,
                                  proposal_boxes,
                                  num_proposals,
                                  image_shapes,
                                  mask_predictions=None,
                                  orig_image_shapes=None):
    """Converts predictions from the second stage box classifier to detections.

    Args:
      refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
      class_predictions_with_background: a 2-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.num_queries, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
      num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
      image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
      mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4] in normalized co-ordinates.
        `detection_scores`: [batch, max_detections]
        `detection_multiclass_scores`: [batch, max_detections,
          num_classes_with_background] tensor with class score distribution for
          post-processed detection boxes including background class if any.
        `detection_anchor_indices`: [batch, max_detections] with anchor
          indices.
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
          (optional) [batch, max_detections, mask_height, mask_width]. Note
          that a pixel-wise sigmoid score converter is applied to the detection
          masks.
        `raw_detection_boxes`: [batch, total_detections, 4] tensor with decoded
          detection boxes in normalized coordinates, before Non-Max Suppression.
          The value total_detections is the number of second stage anchors
          (i.e. the total number of boxes before NMS).
        `raw_detection_scores`: [batch, total_detections,
          num_classes_with_background] tensor of multi-class scores for
          raw detection boxes. The value total_detections is the number of
          second stage anchors (i.e. the total number of boxes before NMS).
    """
    clip_window = self._compute_clip_window(image_shapes)
    refined_box_encodings_batch = tf.reshape(
        refined_box_encodings,
        [-1,
        self.num_queries,
        self._box_coder.code_size])
    class_predictions_with_background_batch = tf.reshape(
        class_predictions_with_background,
        [-1, self.num_queries, self.num_classes + 1]
    )
    refined_decoded_boxes_batch = tf.squeeze(self._batch_decode_boxes(
        tf.expand_dims(refined_box_encodings_batch, axis=2), proposal_boxes), axis=2)
    refined_decoded_boxes_batch = ops.normalized_to_image_coordinates(refined_decoded_boxes_batch, image_shape=orig_image_shapes, temp=True)
    class_predictions_with_background_batch_normalized = self._second_stage_score_conversion_fn(class_predictions_with_background_batch) 
    class_predictions_batch = tf.reshape(class_predictions_with_background_batch_normalized, [-1, self.num_queries, self.num_classes + 1])

    batch_size = shape_utils.combined_static_and_dynamic_shape(
        refined_box_encodings_batch)[0]
    batch_anchor_indices = tf.tile(
        tf.expand_dims(tf.range(self.num_queries), 0),
        multiples=[batch_size, 1])
    additional_fields = {
        'multiclass_scores': tf.slice(class_predictions_with_background_batch_normalized,
                 [0, 0, 1], [-1, -1, -1])
    }

    nmsed_boxes = refined_decoded_boxes_batch
    nmsed_classes = tf.argmax(class_predictions_batch, axis=2)
    nmsed_scores = tf.math.reduce_max(class_predictions_batch, axis=2)

    non_background_mask = tf.cast(tf.greater_equal(nmsed_classes, 1), tf.float32)
    nmsed_boxes = tf.multiply(tf.repeat(tf.expand_dims(non_background_mask, axis=2), axis=2, repeats=4), nmsed_boxes)
    nmsed_classes = tf.cast(nmsed_classes, dtype=tf.float32) - tf.cast(tf.ones_like(nmsed_classes), dtype=tf.float32)# tf.multiply(, non_background_mask)
    nmsed_scores = tf.multiply(nmsed_scores, non_background_mask)

    print("NMSED")
    print(nmsed_boxes)
    print(nmsed_classes)
    print(nmsed_scores)

    nmsed_boxes = shape_utils.static_or_dynamic_map_fn(self._clip_window_prune_boxes, [nmsed_boxes, clip_window])

    print("AFTER MAP", nmsed_boxes)

    detections = {
        fields.DetectionResultFields.detection_boxes:
            nmsed_boxes,
        fields.DetectionResultFields.detection_scores:
            nmsed_scores,
        fields.DetectionResultFields.detection_classes:
            nmsed_classes,
        fields.DetectionResultFields.detection_multiclass_scores:
            additional_fields['multiclass_scores'],
        fields.DetectionResultFields.detection_anchor_indices:
            batch_anchor_indices,
        fields.DetectionResultFields.num_detections:
            tf.cast(tf.count_nonzero(nmsed_scores), dtype=tf.float32),
        fields.DetectionResultFields.raw_detection_boxes:
            refined_box_encodings_batch,
        fields.DetectionResultFields.raw_detection_scores:
            class_predictions_with_background_batch_normalized
    }
    return detections

  def restore_from_classification_checkpoint_fn(
        self,
        first_stage_feature_extractor_scope,
        second_stage_feature_extractor_scope):
      """Returns a map of variables to load from a foreign checkpoint.

      Args:
        first_stage_feature_extractor_scope: A scope name for the first stage
          feature extractor.
        second_stage_feature_extractor_scope: A scope name for the second stage
          feature extractor.

      Returns:
        A dict mapping variable names (to load from a checkpoint) to variables in
        the model graph.
      """
      variables_to_restore = {}
      for variable in variables_helper.get_global_variables_safely():
        for scope_name in [first_stage_feature_extractor_scope]:
          if variable.op.name.startswith(scope_name):
            var_name = variable.op.name.replace(scope_name + '/', '')
            variables_to_restore[var_name] = variable
      return variables_to_restore

  def _batch_decode_boxes(self, box_encodings, anchor_boxes):
    """Decodes box encodings with respect to the anchor boxes.

    Args:
      box_encodings: a 4-D tensor with shape
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        representing box encodings.
      anchor_boxes: [batch_size, num_anchors, self._box_coder.code_size]
        representing decoded bounding boxes. If using a shared box across
        classes the shape will instead be
        [total_num_proposals, 1, self._box_coder.code_size].

    Returns:
      decoded_boxes: a
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        float tensor representing bounding box predictions (for each image in
        batch, proposal and class). If using a shared box across classes the
        shape will instead be
        [batch_size, num_anchors, 1, self._box_coder.code_size].
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    num_classes = combined_shape[2]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode(
        tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
        tiled_anchors_boxlist)
    return tf.reshape(decoded_boxes.get(),
                      tf.stack([combined_shape[0], combined_shape[1],
                                num_classes, 4]))

  def _compute_clip_window(self, image_shapes):
    """Computes clip window for non max suppression based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.cast(
        tf.stack([
            tf.zeros_like(clip_heights),
            tf.zeros_like(clip_heights), clip_heights, clip_widths
        ],
                 axis=1),
        dtype=tf.float32)
    return clip_window

  def _normalize_and_clip_boxes(self, boxes_and_image_shape):
    """Normalize and clip boxes."""
    boxes_per_image = boxes_and_image_shape[0]
    image_shape = boxes_and_image_shape[1]

    boxes_contains_classes_dim = boxes_per_image.shape.ndims == 3
    if boxes_contains_classes_dim:
      boxes_per_image = shape_utils.flatten_first_n_dimensions(
          boxes_per_image, 2)
    normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
        box_list.BoxList(boxes_per_image),
        image_shape[0],
        image_shape[1],
        check_range=False).get()

    normalized_boxes_per_image = box_list_ops.clip_to_window(
        box_list.BoxList(normalized_boxes_per_image),
        tf.constant([0.0, 0.0, 1.0, 1.0], tf.float32),
        filter_nonoverlapping=False).get()

    if boxes_contains_classes_dim:
      max_num_proposals, num_classes, _ = (
          shape_utils.combined_static_and_dynamic_shape(
              boxes_and_image_shape[0]))
      normalized_boxes_per_image = shape_utils.expand_first_dimension(
          normalized_boxes_per_image, [max_num_proposals, num_classes])

    return normalized_boxes_per_image

  def change_coordinate_frame(self, boxlist_window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
      boxlist: A BoxList object holding N boxes.
      window: A rank 1 tensor [4].
      scope: name scope.

    Returns:
      Returns a BoxList object with N boxes.
    """
    boxlist = box_list.BoxList(boxlist_window[0])
    window = boxlist_window[1]
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
      win_height = window[2] - window[0]
      win_width = window[3] - window[1]
      boxlist_new = box_list_ops.scale(box_list.BoxList(
          boxlist.get() - [window[0], window[1], window[0], window[1]]),
                          1.0 / win_height, 1.0 / win_width)
      boxlist_new = box_list_ops._copy_extra_fields(boxlist_new, boxlist)
      return boxlist_new.get()

  def _clip_window_prune_boxes(self, sorted_boxesandclip_window, pad_to_max_output_size=True,
                             change_coordinate_frame=True):
    """Prune boxes with zero area.

    Args:
      sorted_boxes: A BoxList containing k detections.
      clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
        representing the window to clip and normalize boxes to before performing
        non-max suppression.
      pad_to_max_output_size: flag indicating whether to pad to max output size or
        not.
      change_coordinate_frame: Whether to normalize coordinates after clipping
        relative to clip_window (this can only be set to True if a clip_window is
        provided).

    Returns:
      sorted_boxes: A BoxList containing k detections after pruning.
      num_valid_nms_boxes_cumulative: Number of valid NMS boxes
    """
    sorted_boxes = box_list.BoxList(sorted_boxesandclip_window[0])
    clip_window = sorted_boxesandclip_window[1]
    sorted_boxes = box_list_ops.clip_to_window(
        sorted_boxes,
        clip_window,
        filter_nonoverlapping=not pad_to_max_output_size)
    # Set the scores of boxes with zero area to -1 to keep the default
    # behaviour of pruning out zero area boxes.
    #sorted_boxes_size = tf.shape(sorted_boxes.get())[0]
    #non_zero_box_area = tf.cast(box_list_ops.area(sorted_boxes), tf.bool)
    #sorted_boxes_scores = tf.where(
    #    non_zero_box_area, sorted_boxes.get_field(fields.BoxListFields.scores),
    #    -1 * tf.ones(sorted_boxes_size))
    #sorted_boxes.add_field(fields.BoxListFields.scores, sorted_boxes_scores)
    #num_valid_nms_boxes_cumulative = tf.reduce_sum(
    #    tf.cast(tf.greater_equal(sorted_boxes_scores, 0), tf.int32))
    #sorted_boxes = box_list_ops.sort_by_field(sorted_boxes,
    #                                          fields.BoxListFields.scores)
    if change_coordinate_frame:
      sorted_boxes = box_list_ops.change_coordinate_frame(sorted_boxes,
                                                          clip_window)
    return sorted_boxes.get()