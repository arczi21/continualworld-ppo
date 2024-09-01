"""Core functions of the PPO algorithm."""
from typing import Callable, Iterable, List, Tuple

import gym
import numpy as np
import scipy.signal
import tensorflow as tf

from continualworld.envs import MW_ACT_LEN, MW_OBS_LEN
from tensorflow.keras import Input, Model


EPS = 1e-8

LOG_STD_MAX = 0.41
LOG_STD_MIN = -0.7


def distribute_value(value, num_proc):
    """Adjusts training parameters for distributed training.

    In case of distributed training frequencies expressed in global steps have
    to be adjusted to local steps, thus divided by the number of processes.
    """
    return max(value // num_proc, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[
           ::-1]


@tf.function
def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
            ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
            2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def mlp(
    input_dim: int, hidden_sizes: Iterable[int], activation: Callable, layer_norm: bool = False
) -> Model:
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(hidden_sizes[0]))
    if layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(tf.nn.tanh))
    else:
        model.add(tf.keras.layers.Activation(activation))
    for size in hidden_sizes[1:]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    return model

def _choose_head(out: tf.Tensor, obs: tf.Tensor, num_heads: int) -> tf.Tensor:
    """For multi-head output, choose appropriate head.

    We assume that task number is one-hot encoded as a part of observation.

    Args:
      out: multi-head output tensor from the model
      obs: obsevation batch. We assume that last num_heads dims is one-hot encoding of task
      num_heads: number of heads

    Returns:
      tf.Tensor: output for the appropriate head
    """

    batch_size = tf.shape(obs)[0]
    out = tf.reshape(out, [tf.shape(out)[0], -1, num_heads])
    obs = tf.reshape(obs[:, -num_heads:], [batch_size, num_heads, 1])
    return tf.squeeze(out @ obs, axis=2)

class MlpActor(Model):
    def __init__(
        self,
        input_dim: int,
        action_space: gym.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpActor, self).__init__()
        self.num_heads = num_heads
        # if True, one-hot encoding of the task will not be appended to observation.
        self.hide_task_id = hide_task_id

        if self.hide_task_id:
            input_dim = MW_OBS_LEN

        self.core = mlp(input_dim, hidden_sizes, activation, layer_norm=layer_norm)
        self.head_mu = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                tf.keras.layers.Dense(action_space.shape[0] * num_heads),
            ]
        )
        self.log_std_dim = action_space.shape[0] * num_heads
        self._log_std = tf.Variable(
            initial_value=-0.5 * np.ones(shape=(1, self.log_std_dim),
                                         dtype=np.float32), trainable=True,
            name='log_std_dev')
        self.action_space = action_space

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        obs = x
        if self.hide_task_id:
            x = x[:, :MW_OBS_LEN]
        x = self.core(x)
        mu = self.head_mu(x)
        log_std = tf.clip_by_value(self._log_std, LOG_STD_MIN, LOG_STD_MAX)

        if self.num_heads > 1:
            mu = _choose_head(mu, obs, self.num_heads)
            log_std = _choose_head(self._log_std, obs, self.num_heads)

        return mu, log_std

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return (
                self.core.trainable_variables
                + self.head_mu.trainable_variables
                + [self._log_std]
            )

    @tf.function
    def action(self, observations, deterministic=False):
        mu, log_std = self(observations)
        std = tf.exp(log_std)
        if deterministic:
            return mu
        else:
            return mu + tf.random.normal(tf.shape(input=mu)) * std

    @tf.function
    def action_logprob(self, observations, actions):
        mu, log_std = self(observations)
        return gaussian_likelihood(actions, mu, log_std)

class MlpCritic(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpCritic, self).__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = (
            num_heads  # if True, one-hot encoding of the task will not be appended to observation.
        )

        if self.hide_task_id:
            input_dim = MW_OBS_LEN
        self.core = mlp(input_dim, hidden_sizes, activation, layer_norm=layer_norm)
        self.head = tf.keras.Sequential(
            [Input(shape=(hidden_sizes[-1],)), tf.keras.layers.Dense(num_heads)]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        obs = x
        if self.hide_task_id:
            x = x[:, :MW_OBS_LEN]
        x = self.head(self.core(x))
        if self.num_heads > 1:
            x = _choose_head(x, obs, self.num_heads)
        x = tf.squeeze(x, axis=1)
        return x

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return self.core.trainable_variables + self.head.trainable_variables





