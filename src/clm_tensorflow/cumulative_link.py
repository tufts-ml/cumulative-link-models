#!/usr/bin/env python3
"""The cumulative link distribution class."""

import warnings
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability import layers as tfpl
from tensorflow_probability import math as tfp_math
from tensorflow_probability import util
from tensorflow_probability.python.internal import dtype_util, tensor_util, prefer_static, samplers, tensorshape_util
from tensorflow_probability.python.distributions.ordered_logistic import _broadcast_cat_event_and_params


class CumulativeLink(tfd.Distribution):
    def __init__(
        self,
        cutpoints: tf.Tensor,
        loc: tf.Tensor,
        link: str = 'probit',
        scale: tf.float32 = 1.,
        dtype=tf.int32,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = 'CumulativeLink',
    ) -> None:
        """Initialize Cumulative Link distribution.

        `C` : Number of desired ordered classes

        Parameters
        ----------
        cutpoints : `tf.Tensor`
            A floating-point `Tensor` that is a `C-1`-length vector of cutpoint
            values. The vector of cutpoints should be non-decreasing, which is 
            only checked if `validate_args=True`.
        loc : `tf.Tensor`
            A floating-point `Tensor`. The entries represent the
            mean(s) of the latent link distribution(s), which can also be
            considered the network outputs (or logits).
        link : `str`, optional
            Desired link function to use to model the noise of the latent
            function, by default 'probit'. Must be be either `'probit'`
            or `'logit'`.
        scale : `tf.float32`, optional
            Desired scale (stddev) of the noise applied on the latent function, 
            by default `1`.
        dtype : optional
            The type of the event samples, by default `tf.int32`
        validate_args : bool, optional
            When `True` distribution parameters are checked for validity despite
            possibly degrading runtime performance. When `False` invalid inputs 
            may silently render incorrect outputs, by default `False`.
        allow_nan_stats : `bool`, optional
            When `True`, statistics (e.g. mode) use the value `NaN` to indicate 
            the result is undefined. When `False`, an exception is raised if 
            one or more of the statistic's batch members are undefined, 
            by default True
        name : str, optional
            Python `str` name prefixed to Ops created by this class, 
            by default 'CumulativeLink'
        """

        parameters = dict(locals())

        with tf.name_scope(name) as name:

            # Cumulative link specific parameters
            float_dtype = dtype_util.common_dtype(
                [cutpoints, loc, scale],
                dtype_hint=tf.float32,
            )
            self._cutpoints = tensor_util.convert_nonref_to_tensor(
                cutpoints, dtype_hint=float_dtype, name='cutpoints')
            self._loc = tensor_util.convert_nonref_to_tensor(
                loc, dtype_hint=float_dtype, name='loc')
            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, dtype_hint=float_dtype, name='scale')

            # Define the link function
            if link == 'probit':
                self._link = tfd.Normal(loc=0, scale=1)
            elif link == 'logit':
                self._link = tfd.Logistic(loc=0, scale=1)
            else:
                raise ValueError(
                    'The link argument must be either "probit" or "logit".')

            super().__init__(
                dtype=dtype,
                reparameterization_type=tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name
            )

    @classmethod
    def _params_event_ndims(cls):
        return dict(cutpoints=1, loc=0)

    @staticmethod
    def _param_shapes(sample_shape, num_classes=None):
        """Shapes of class parameters depending on desired sample shape 
        and number of classes `C`."""
        return {
            'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
            'cutpoints': tf.convert_to_tensor(num_classes-1),
            'scale': tf.convert_to_tensor(1),
        }

    @property
    def cutpoints(self):
        """Input argument `cutpoints`."""
        return self._cutpoints

    @property
    def loc(self):
        """Input argument `loc`."""
        return self._loc

    @property
    def scale(self):
        """Input argument `scale`."""
        return self._scale

    @property
    def link(self):
        """Distribution class argument `link`."""
        return self._link

    def categorical_log_probs(self):
        """Log probabilities for the `C` ordered class categories."""
        z_values = self._z_values()
        log_cdfs = self.link.log_cdf(z_values)
        return tfp_math.log_sub_exp(log_cdfs[..., :-1], log_cdfs[..., 1:])

    def categorical_probs(self):
        """Probabilities for the `C` ordered class categories."""
        return tf.math.exp(self.categorical_log_probs())

    def _z_values(self, eps=1e-8) -> tf.Tensor:
        """`z` values used to compute CDFs for smooth likelihood.

        Replaces all occurances of `inf` with `1/eps` to avoide over/underflow issues.
        """
        z = (self._augmented_cutpoints() -
             self.loc[..., tf.newaxis]) / self.scale
        # replace inf/-inf with very large or very small value
        z = tf.where(tf.math.is_inf(z), tf.sign(z) * (1/eps), z)
        return z

    def _log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """Log probability mass function. 

        Returns the log probabilities given some known ordinal classes.

        Parameters
        ----------
        x : tf.Tensor
            Known ordinal classes

        Returns
        -------
        log_prob : tf.Tensor
            Log probabilities of specified ordinal classes given defined 
            cumulative link distribution.
        """
        num_categories = self._num_categories()
        z = self._z_values()
        x, augmented_log_cdf = _broadcast_cat_event_and_params(
            event=x,
            params=self.link.log_cdf(z),
            base_dtype=dtype_util.base_dtype(self.dtype))
        x_flat = tf.reshape(x, [-1, 1])
        augmented_log_cdf_flat = tf.reshape(
            augmented_log_cdf, [-1, num_categories + 1])
        log_cdf_flat_xm1 = tf.gather(
            params=augmented_log_cdf_flat,
            indices=tf.clip_by_value(x_flat, 0, num_categories),
            batch_dims=1)
        log_cdf_flat_x = tf.gather(
            params=augmented_log_cdf_flat,
            indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
            batch_dims=1)
        log_prob_flat = tfp_math.log_sub_exp(
            log_cdf_flat_xm1, log_cdf_flat_x)
        # Deal with case where both survival probabilities are -inf, which gives
        # `log_prob_flat = nan` when it should be -inf.
        minus_inf = tf.constant(-np.inf, dtype=log_prob_flat.dtype)
        log_prob_flat = tf.where(
            x_flat > num_categories - 1, minus_inf, log_prob_flat)
        return tf.reshape(log_prob_flat, shape=tf.shape(x))

    def _augmented_cutpoints(self):
        """Augment `cutpoints` with +/- `np.inf`"""
        cutpoints = tf.convert_to_tensor(self.cutpoints)
        inf = tf.fill(
            cutpoints[..., :1].shape,
            tf.constant(np.inf, dtype=cutpoints.dtype))
        return tf.concat([-inf, cutpoints, inf], axis=-1)

    def _num_categories(self):
        """Number of ordinal categories `C`"""
        return tf.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

    def _sample_n(self, n, seed=None):
        """Internal method to help generate `n` samples."""
        logits = tf.reshape(
            self.categorical_log_probs(), [-1, self._num_categories()])
        draws = samplers.categorical(logits, n, dtype=self.dtype, seed=seed)
        return tf.reshape(
            tf.transpose(draws),
            shape=tf.concat([[n], self._batch_shape_tensor()], axis=0))

    def _batch_shape_tensor(self, cutpoints=None, loc=None):
        cutpoints = self.cutpoints if cutpoints is None else cutpoints
        loc = self.loc if loc is None else loc
        return prefer_static.broadcast_shape(
            prefer_static.shape(cutpoints)[:-1],
            prefer_static.shape(loc))

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self.loc.shape, self.cutpoints.shape[:-1])

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _log_cdf(self, x):
        return tfp_math.log1mexp(self._log_cdf_function(x))

    def _log_cdf_function(self, x):
        num_categories = self._num_categories()
        z = self._z_values()
        x, augmented_log_cdf = _broadcast_cat_event_and_params(
            event=x,
            params=self.link.log_cdf(z),
            base_dtype=dtype_util.base_dtype(self.dtype))
        x_flat = tf.reshape(x, [-1, 1])
        augmented_log_cdf_flat = tf.reshape(
            augmented_log_cdf, [-1, num_categories + 1])
        log_cdf_flat = tf.gather(
            params=augmented_log_cdf_flat,
            indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
            batch_dims=1)
        return tf.reshape(log_cdf_flat, shape=tf.shape(x))

    def _entropy(self):
        log_probs = self.categorical_log_probs()
        return -tf.reduce_sum(
            tf.math.multiply_no_nan(log_probs, tf.math.exp(log_probs)),
            axis=-1)

    def _mode(self):
        log_probs = self.categorical_log_probs()
        mode = tf.argmax(log_probs, axis=-1, output_type=self.dtype)
        tensorshape_util.set_shape(mode, log_probs.shape[:-1])
        return mode

    def _mean(self):
        return self._mode()

    def _default_event_space_bijector(self):
        return

    @classmethod
    def param_shapes(cls, sample_shape, num_classes, name='DistributionParamShapes'):
        """Overloaded tfd.Distribution.param_shapes() method to support classes.

        See tfd.Distribition for more info.
        """
        with tf.name_scope(name):
            return cls._param_shapes(sample_shape, num_classes)

    @classmethod
    def param_static_shapes(cls, sample_shape, num_classes):
        """Overloaded tfd.Distribution.param_static_shapes() method to support classes.

        See tfd.Distribition for more info.
        """
        if isinstance(sample_shape, tf.TensorShape):
            if not tensorshape_util.is_fully_defined(sample_shape):
                raise ValueError(
                    'TensorShape sample_shape must be fully defined')
            sample_shape = tensorshape_util.as_list(sample_shape)

        params = cls.param_shapes(sample_shape, num_classes)

        static_params = {}
        for name, shape in params.items():
            static_shape = tf.get_static_value(shape)
            if static_shape is None:
                raise ValueError(
                    'sample_shape must be a fully-defined TensorShape or list/tuple')
            static_params[name] = tf.TensorShape(static_shape)

        return static_params


class SimpleCumulativeLink(tfd.OrderedLogistic):
    def __init__(
        self,
        cutpoints: tf.Tensor,
        loc: tf.Tensor,
        link: str = 'probit',
        scale: tf.float32 = 1.,
        dtype=tf.int32,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = 'SimpleCumulativeLink',
    ) -> None:
        """Initialize Cumulative Link distribution.

        This simplied version of the class inherits from `tfd.OrderedLogistic`,
        not requiring to rewrite many of the required methods for a `Distribution`
        class, but overloading some of the methods to suite the CLM implementation.

        `C` : Number of desired ordered classes

        Parameters
        ----------
        cutpoints : `tf.Tensor`
            A floating-point `Tensor` that is a `C-1`-length vector of cutpoint
            values. The vector of cutpoints should be non-decreasing, which is 
            only checked if `validate_args=True`.
        loc : `tf.Tensor`
            A floating-point `Tensor`. The entries represent the
            mean(s) of the latent link distribution(s), which can also be
            considered the network outputs (or logits).
        link : `str`, optional
            Desired link function to use to model the noise of the latent
            function, by default 'probit'. Must be be either `'probit'`
            or `'logit'`.
        scale : `tf.float32`, optional
            Desired scale (stddev) of the noise applied on the latent function, 
            by default `1`.
        dtype : optional
            The type of the event samples, by default `tf.int32`
        validate_args : bool, optional
            When `True` distribution parameters are checked for validity despite
            possibly degrading runtime performance. When `False` invalid inputs 
            may silently render incorrect outputs, by default `False`.
        allow_nan_stats : `bool`, optional
            When `True`, statistics (e.g. mode) use the value `NaN` to indicate 
            the result is undefined. When `False`, an exception is raised if 
            one or more of the statistic's batch members are undefined, 
            by default True
        name : str, optional
            Python `str` name prefixed to Ops created by this class, 
            by default 'CumulativeLink'
        """
        with tf.name_scope(name) as name:
            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, dtype_hint=tf.float32, name='scale')

            if link == 'probit':
                self._link = tfd.Normal(loc=0, scale=1)
            elif link == 'logit':
                self._link = tfd.Logistic(loc=0, scale=1)
            else:
                raise ValueError(
                    'The link argument must be either "probit" or "logit".')

            super().__init__(
                cutpoints=cutpoints,
                loc=loc,
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name
            )

    @property
    def scale(self):
        """Input argument `scale`."""
        return self._scale

    @property
    def link(self):
        """Distribution class argument `link`."""
        return self._link

    def _z_values(self, eps=1e-8) -> tf.Tensor:
        """`z` values used to compute CDFs for smoother likelihood.

        Replaces all occurances of `inf` with `1/eps` to avoide over/underflow issues.
        """
        z = (self._augmented_cutpoints() -
             self.loc[..., tf.newaxis]) / self.scale
        # replace inf/-inf with very large or very small value
        z = tf.where(tf.math.is_inf(z), tf.sign(z) * (1/eps), z)
        return z

    def categorical_log_probs(self):
        """Log probabilities for the `C` ordered categories."""
        z_values = self._z_values()
        log_cdfs = self.link.log_cdf(z_values)
        return tfp_math.log_sub_exp(log_cdfs[..., :-1], log_cdfs[..., 1:])

    def _log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """Log probability mass function. 

        Returns the log probabilities given some known ordinal classes.

        Parameters
        ----------
        x : tf.Tensor
            Known ordinal classes

        Returns
        -------
        log_prob : tf.Tensor
            Log probabilities of specified ordinal classes given defined 
            cumulative link distribution.
        """
        num_categories = self._num_categories()
        z = self._z_values()
        x, augmented_log_cdf = _broadcast_cat_event_and_params(
            event=x,
            params=self.link.log_cdf(z),
            base_dtype=dtype_util.base_dtype(self.dtype))
        x_flat = tf.reshape(x, [-1, 1])
        augmented_log_cdf_flat = tf.reshape(
            augmented_log_cdf, [-1, num_categories + 1])
        log_cdf_flat_xm1 = tf.gather(
            params=augmented_log_cdf_flat,
            indices=tf.clip_by_value(x_flat, 0, num_categories),
            batch_dims=1)
        log_cdf_flat_x = tf.gather(
            params=augmented_log_cdf_flat,
            indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
            batch_dims=1)
        log_prob_flat = tfp_math.log_sub_exp(
            log_cdf_flat_xm1, log_cdf_flat_x)
        # Deal with case where both survival probabilities are -inf, which gives
        # `log_prob_flat = nan` when it should be -inf.
        minus_inf = tf.constant(-np.inf, dtype=log_prob_flat.dtype)
        log_prob_flat = tf.where(
            x_flat > num_categories - 1, minus_inf, log_prob_flat)
        return tf.reshape(log_prob_flat, shape=tf.shape(x))

    def _mean(self):
        """Mean of distribution--treated as mode."""
        return self._mode()

    @staticmethod
    def _param_shapes(sample_shape, num_classes):
        """Shapes of class parameters depending on desired sample shape 
        and number of classes `C`."""
        return {
            'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
            'cutpoints': tf.convert_to_tensor(num_classes-1),
            'scale': tf.convert_to_tensor(1),
        }

    @classmethod
    def param_shapes(cls, sample_shape, num_classes, name='DistributionParamShapes'):
        """Overloaded tfd.Distribution.param_shapes() method to support classes.

        See tfd.Distribition for more info.
        """
        with tf.name_scope(name):
            return cls._param_shapes(sample_shape, num_classes)

    @classmethod
    def param_static_shapes(cls, sample_shape, num_classes):
        """Overloaded tfd.Distribution.param_static_shapes() method to support classes.

        See tfd.Distribition for more info.
        """
        if isinstance(sample_shape, tf.TensorShape):
            if not tensorshape_util.is_fully_defined(sample_shape):
                raise ValueError(
                    'TensorShape sample_shape must be fully defined')
            sample_shape = tensorshape_util.as_list(sample_shape)

        params = cls.param_shapes(sample_shape, num_classes)

        static_params = {}
        for name, shape in params.items():
            static_shape = tf.get_static_value(shape)
            if static_shape is None:
                raise ValueError(
                    'sample_shape must be a fully-defined TensorShape or list/tuple')
            static_params[name] = tf.TensorShape(static_shape)

        return static_params
