# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Bipartite matcher implementation."""

import tensorflow.compat.v1 as tf

from object_detection.core import matcher
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianBipartiteMatcher(matcher.Matcher):
  """Wraps a Tensorflow greedy bipartite matcher."""

  def __init__(self, use_matmul_gather=False):
    """Constructs a Matcher.

    Args:
      use_matmul_gather: Force constructed match objects to use matrix
        multiplication based gather instead of standard tf.gather.
        (Default: False).
    """
    super(HungarianBipartiteMatcher, self).__init__(
        use_matmul_gather=use_matmul_gather)

  def _match(self, similarity_matrix, valid_rows):
    """Bipartite matches a collection rows and columns. A greedy bi-partite.

    TODO(rathodv): Add num_valid_columns options to match only that many columns
    with all the rows.O

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      valid_rows: A boolean tensor of shape [N] indicating the rows that are
        valid.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    valid_row_sim_matrix = tf.gather(similarity_matrix,
                                     tf.squeeze(tf.where(valid_rows), axis=-1))
    distance_matrix = -1 * valid_row_sim_matrix
    num_valid_rows = tf.reduce_sum(tf.cast(valid_rows, dtype=tf.float32))
    #numpy_distance = distance_matrix.numpy()
    #print(numpy_distance)
    
    row_indices, col_indices = tf.autograph.experimental.do_not_convert(
                                                 tf.numpy_function(func=linear_sum_assignment,
                                                 inp=[distance_matrix],
                                                 Tout=tf.float32))
    print("DID IT")
    
    #match_results = np.full(numpy_distance.shape[1], -1)
    
    #for i in range(len(col_indices)):
    #    match_results[col_indices[i]] = row_indices[i] 

    #match_results = tf.convert_to_tensor(match_results)
    match_results = tf.reshape(match_results, [-1])
    match_results = tf.cast(match_results, tf.int32)

    return match_results