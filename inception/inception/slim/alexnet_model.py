# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def alexnet(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for op_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.op_scope([inputs], scope, 'alexnet'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      # conv and pool will do padding
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            padding='SAME'):
        # define the initial distribution of filter weight
        with scopes.arg_scope([ops.conv2d], stddev=0.01):
          end_points['conv1'] = ops.conv2d(inputs, 96, [11, 11], stride=4,
                                           scope='conv1')
          end_points['pool1'] = ops.max_pool(end_points['conv1'], [3, 3],
                                             stride=2, scope='pool1')
          end_points['conv2'] = ops.conv2d(end_points['pool1'], 256, [5, 5],
                                           bias=1.0, scope='conv2')
          end_points['pool2'] = ops.max_pool(end_points['conv2'], [3, 3],
                                             stride=2, scope='pool2')
          end_points['conv3'] = ops.conv2d(end_points['pool2'], 384, [3, 3],
                                           scope='conv3')
          end_points['conv4'] = ops.conv2d(end_points['conv3'], 384, [3, 3],
                                           bias=1.0, scope='conv4')
          end_points['conv5'] = ops.conv2d(end_points['conv4'], 256, [3, 3],
                                           bias=1.0, scope='conv5')
          end_points['pool5'] = ops.max_pool(end_points['conv5'], [3, 3],
                                             stride=2, scope='pool5')

      # reshape the 4d tensor into 2d
      end_points['flatten'] = ops.flatten(end_points['pool5'], scope='flatten')

      # define the initial distribution of fc weight
      with scopes.arg_scope([ops.fc], stddev=0.005, bias=1.0):
        # define the dropout ratio
        with scopes.arg_scope([ops.dropout], keep_prob=dropout_keep_prob):
          end_points['fc6'] = ops.fc(end_points['flatten'], 4096, scope='fc6')
          end_points['drop6'] = ops.dropout(end_points['fc6'], scope='drop6')
          end_points['fc7'] = ops.fc(end_points['drop6'], 4096, scope='fc7')
          end_points['drop7'] = ops.dropout(end_points['fc7'], scope='drop7')
          end_points['fc8'] = ops.fc(end_points['drop7'], num_classes,
                                     activation=None,
                                     scope='fc8', restore=restore_logits)
      return end_points['fc8'], end_points


def alexnet_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope