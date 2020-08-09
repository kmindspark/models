# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/detr.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import anchor_generator_pb2 as object__detection_dot_protos_dot_anchor__generator__pb2
from object_detection.protos import box_predictor_pb2 as object__detection_dot_protos_dot_box__predictor__pb2
from object_detection.protos import hyperparams_pb2 as object__detection_dot_protos_dot_hyperparams__pb2
from object_detection.protos import image_resizer_pb2 as object__detection_dot_protos_dot_image__resizer__pb2
from object_detection.protos import losses_pb2 as object__detection_dot_protos_dot_losses__pb2
from object_detection.protos import post_processing_pb2 as object__detection_dot_protos_dot_post__processing__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/detr.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_pb=_b('\n\"object_detection/protos/detr.proto\x12\x17object_detection.protos\x1a.object_detection/protos/anchor_generator.proto\x1a+object_detection/protos/box_predictor.proto\x1a)object_detection/protos/hyperparams.proto\x1a+object_detection/protos/image_resizer.proto\x1a$object_detection/protos/losses.proto\x1a-object_detection/protos/post_processing.proto\"\xbe\x03\n\x04\x44\x45TR\x12\x13\n\x0bnum_classes\x18\x01 \x01(\x05\x12<\n\rimage_resizer\x18\x02 \x01(\x0b\x32%.object_detection.protos.ImageResizer\x12H\n\x11\x66\x65\x61ture_extractor\x18\x03 \x01(\x0b\x32-.object_detection.protos.DETRFeatureExtractor\x12(\n\x1dgiou_localization_loss_weight\x18\x04 \x01(\x02:\x01\x31\x12&\n\x1bl1_localization_loss_weight\x18\x05 \x01(\x02:\x01\x31\x12%\n\x1a\x63lassification_loss_weight\x18\x06 \x01(\x02:\x01\x31\x12+\n\x1cuse_matmul_gather_in_matcher\x18\x07 \x01(\x08:\x05\x66\x61lse\x12@\n\x0fpost_processing\x18\x08 \x01(\x0b\x32\'.object_detection.protos.PostProcessing\x12\x18\n\x0bnum_queries\x18\t \x01(\x05:\x03\x31\x30\x30\x12\x17\n\nhidden_dim\x18\n \x01(\x05:\x03\x32\x35\x36\"I\n\x14\x44\x45TRFeatureExtractor\x12\x0c\n\x04type\x18\x01 \x01(\t\x12#\n\x14\x62\x61tch_norm_trainable\x18\x02 \x01(\x08:\x05\x66\x61lse')
  ,
  dependencies=[object__detection_dot_protos_dot_anchor__generator__pb2.DESCRIPTOR,object__detection_dot_protos_dot_box__predictor__pb2.DESCRIPTOR,object__detection_dot_protos_dot_hyperparams__pb2.DESCRIPTOR,object__detection_dot_protos_dot_image__resizer__pb2.DESCRIPTOR,object__detection_dot_protos_dot_losses__pb2.DESCRIPTOR,object__detection_dot_protos_dot_post__processing__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DETR = _descriptor.Descriptor(
  name='DETR',
  full_name='object_detection.protos.DETR',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='object_detection.protos.DETR.num_classes', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_resizer', full_name='object_detection.protos.DETR.image_resizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='object_detection.protos.DETR.feature_extractor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='giou_localization_loss_weight', full_name='object_detection.protos.DETR.giou_localization_loss_weight', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='l1_localization_loss_weight', full_name='object_detection.protos.DETR.l1_localization_loss_weight', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='classification_loss_weight', full_name='object_detection.protos.DETR.classification_loss_weight', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_matmul_gather_in_matcher', full_name='object_detection.protos.DETR.use_matmul_gather_in_matcher', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='post_processing', full_name='object_detection.protos.DETR.post_processing', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_queries', full_name='object_detection.protos.DETR.num_queries', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hidden_dim', full_name='object_detection.protos.DETR.hidden_dim', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=330,
  serialized_end=776,
)


_DETRFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='DETRFeatureExtractor',
  full_name='object_detection.protos.DETRFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='object_detection.protos.DETRFeatureExtractor.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_norm_trainable', full_name='object_detection.protos.DETRFeatureExtractor.batch_norm_trainable', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=778,
  serialized_end=851,
)

_DETR.fields_by_name['image_resizer'].message_type = object__detection_dot_protos_dot_image__resizer__pb2._IMAGERESIZER
_DETR.fields_by_name['feature_extractor'].message_type = _DETRFEATUREEXTRACTOR
_DETR.fields_by_name['post_processing'].message_type = object__detection_dot_protos_dot_post__processing__pb2._POSTPROCESSING
DESCRIPTOR.message_types_by_name['DETR'] = _DETR
DESCRIPTOR.message_types_by_name['DETRFeatureExtractor'] = _DETRFEATUREEXTRACTOR

DETR = _reflection.GeneratedProtocolMessageType('DETR', (_message.Message,), dict(
  DESCRIPTOR = _DETR,
  __module__ = 'object_detection.protos.detr_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.DETR)
  ))
_sym_db.RegisterMessage(DETR)

DETRFeatureExtractor = _reflection.GeneratedProtocolMessageType('DETRFeatureExtractor', (_message.Message,), dict(
  DESCRIPTOR = _DETRFEATUREEXTRACTOR,
  __module__ = 'object_detection.protos.detr_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.DETRFeatureExtractor)
  ))
_sym_db.RegisterMessage(DETRFeatureExtractor)


# @@protoc_insertion_point(module_scope)
