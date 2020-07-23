# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.nlp.modeling.layers import position_embedding
from object_detection.meta_architectures import detr_attention as attention_layer
from official.nlp.transformer import ffn_layer
from official.nlp.transformer import model_utils
from official.nlp.transformer.utils.tokenizer import EOS_ID
from official.modeling import tf_utils

import math

class Transformer(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in the paper: End-to-End Object Detection with Transformers

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.1,
               layer_postprocess_dropout=0.1, relu_dropout=0.1, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32, name="ODTransformer"):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(Transformer, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self._dtype = tf.float32
    self._encoder_stack = EncoderStack(self._hidden_size,
                                       self._num_heads,
                                       self._attention_dropout,
                                       self._layer_postprocess_dropout,
                                       self._relu_dropout,
                                       self._filter_size,
                                       self._num_hidden_layers)
    self._decoder_stack = DecoderStack(self._hidden_size,
                                       self._num_heads,
                                       self._attention_dropout,
                                       self._layer_postprocess_dropout,
                                       self._relu_dropout,
                                       self._filter_size,
                                       self._num_hidden_layers)
    self._position_embedding = TwoDimensionalPositionEmbedding(
        hidden_size=self._hidden_size)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers,
        "_dtype": self._dtype,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item (optional), targets: None or int tensor with shape
          [batch_size, target_length].
      training: boolean, whether in training mode or not.

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: [batch_size, decoded length]
          scores: [batch_size, float]}
      Even when float16 is used, the output tensor(s) are always float32.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    inputs, targets = inputs[0], inputs[1]

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs, spatial = self.encode(inputs, training)
      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      logits = self.decode(targets, encoder_outputs, training, encoding=spatial)
      return logits

  def encode(self, encoder_inputs, training):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.

      with tf.name_scope("add_pos_encoding"):
        pos_encoding = self._position_embedding(inputs=encoder_inputs)
        pos_encoding = tf.cast(pos_encoding, self._dtype)

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self._layer_postprocess_dropout)

      return self._encoder_stack(
          encoder_inputs, training=training, encoding=pos_encoding), pos_encoding

  def decode(self, targets, encoder_outputs, training, encoding=None):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = tf.cast(targets, self._dtype)
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(decoder_inputs,
                                [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      #with tf.name_scope("add_pos_encoding"):
      #  
        #pos_encoding = self.position_embedding(decoder_inputs)
        #pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
      #  decoder_inputs += pos_encoding
      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self._layer_postprocess_dropout)

      # Run values
      outputs = self._decoder_stack(
          decoder_inputs,
          encoder_outputs,
          training=training,
          encoding=encoding,
          queries=decoder_inputs)
      return outputs

class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, layer_postprocess_dropout):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self._postprocess_dropout = layer_postprocess_dropout

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "_postprocess_dropout": self._postprocess_dropout,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer(*args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self._postprocess_dropout)
    return self.layer_norm(x + y)

class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.1,
               layer_postprocess_dropout=0.1, relu_dropout=0.1, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32):
    super(EncoderStack, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    for _ in range(self._num_hidden_layers):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          self._hidden_size, self._filter_size, self._relu_dropout)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapper(feed_forward_network, self._layer_postprocess_dropout)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers,
        "_dtype": self._dtype,
    }

  def call(self, encoder_inputs, training, encoding=None):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    print(self.layers)
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs,
              encoder_inputs + encoding,
              encoder_inputs,
              training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.1,
               layer_postprocess_dropout=0.1, relu_dropout=0.1, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32):
    super(DecoderStack, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self.layers = []

  def build(self, input_shape):
    """Builds the decoder stack."""
    for _ in range(self._num_hidden_layers):
      self_attention_layer = attention_layer.SelfAttention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      enc_dec_attention_layer = attention_layer.Attention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          self._hidden_size, self._filter_size, self._relu_dropout)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapper(enc_dec_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapper(feed_forward_network, self._layer_postprocess_dropout)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(DecoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers,
        "_dtype": self._dtype,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           training,
           cache=None,
           decode_loop_step=None,
           encoding=None,
           queries=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape
        [batch_size, target_length, hidden_size].
      encoder_outputs: A tensor with shape
        [batch_size, input_length, hidden_size]
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_inputs + queries,
              decoder_inputs,
              training=training,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              decoder_inputs + queries,
              encoder_outputs + encoding,
              encoder_outputs,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)

@tf.keras.utils.register_keras_serializable(package="Text")
class TwoDimensionalPositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized in
   "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Arguments:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
  """

  def __init__(self,
               hidden_size,
               min_timescale=1.0,
               max_timescale=1.0e4,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically
    # unstable in float16.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super(TwoDimensionalPositionEmbedding, self).__init__(**kwargs)
    self._hidden_size = hidden_size / 2
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale,
        "length": self._length,
    }
    base_config = super(TwoDimensionalPositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_1d_encoding(self, length):
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = self._hidden_size // 2
    min_timescale, max_timescale = self._min_timescale, self._max_timescale
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales,
                                                               0)
    position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                                    axis=1)
    return position_embeddings


  def call(self, inputs, length=None):
    """Implements call() for the layer.

    Args:
      inputs: An tensor whose second dimension will be used as `length`. If
        `None`, the other `length` argument must be specified.
      length: An optional integer specifying the number of positions. If both
        `inputs` and `length` are spcified, `length` must be equal to the
        second dimension of `inputs`.

    Returns:
      A tensor in shape of [length, hidden_size].
    """
    input_shape = tf_utils.get_shape_list(inputs)
    per_axis_size = int(math.sqrt(input_shape[1]))
    one_d_encoding = self._get_1d_encoding(per_axis_size)
    encoding_x = tf.repeat(one_d_encoding, repeats=per_axis_size, axis=0)
    encoding_y = tf.tile(one_d_encoding, multiples=[per_axis_size, 1])
    return tf.concat([encoding_x, encoding_y], axis=1)