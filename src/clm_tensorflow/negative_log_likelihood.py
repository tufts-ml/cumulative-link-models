#!/usr/bin/env python3
"""Class to minimize negative log likelihood as loss."""

import tensorflow as tf

class NegativeLogLikelihood(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='neg_log_lik'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)