from typing import List, Tuple

import numpy as np
import tensorflow as tf

from continualworld.methods_ppo.regularization_ppo import Regularization_PPO
from continualworld.ppo.ppo import PPOBuffer



class EWC_PPO(Regularization_PPO):
    """EWC regularization method.

    https://arxiv.org/abs/1612.00796"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @tf.function
    def _get_grads(
        self,
        obs,
        actions,
        advantages,
        returns,
        logps,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph
            mu, log_std = self.actor(obs)

            std = tf.exp(log_std)

            v1 = self.critic(obs)


        # Compute diagonal of the Fisher matrix
        actor_mu_gs = g.jacobian(mu, self.actor_common_variables)
        actor_std_gs = g.jacobian(std, self.actor_common_variables)
        v1_gs = g.jacobian(v1, self.critic.common_variables)
        del g
        return actor_mu_gs, actor_std_gs, v1_gs, std

    def _update_reg_weights(
        self, replay_buffer: PPOBuffer, batches_num: int = 10, batch_size: int = 256
    ) -> None:
        """Calculate importance weights representing how important each weight is for the current
        task."""
        all_weights = []

        for batch_idx in range(batches_num):
            batch_sampled = replay_buffer.sample_batch(batch_size)
            all_weights += [self._get_importance_weights(**batch_sampled)]

        mean_weights = []
        for weights in zip(*all_weights):
            mean_weights += [tf.reduce_mean(tf.stack(weights, 0), 0)]

        self._merge_weights(mean_weights)


    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:

        actor_mu_gs, actor_std_gs, v1_gs, std = self._get_grads(**batch)

        reg_weights = []
        for mu_g, std_g in zip(actor_mu_gs, actor_std_gs):
            if mu_g is None and std_g is None:
                raise ValueError("Both mu and std gradients are None!")
            if mu_g is None:
                mu_g = tf.zeros_like(std_g)
            if std_g is None:
                std_g = tf.zeros_like(mu_g)

            # Broadcasting std for every parameter in the model
            dims_to_add = int(tf.rank(mu_g) - tf.rank(std))
            broad_shape = std.shape + [1] * dims_to_add
            broad_std = tf.reshape(std, broad_shape)  # broadcasting

            # Fisher information, see the derivation
            fisher = 1 / (broad_std ** 2 + 1e-6) * (mu_g ** 2 + 2 * std_g ** 2)

            # Sum over the output dimensions
            fisher = tf.reduce_sum(fisher, 1)

            # Clip from below
            fisher = tf.clip_by_value(fisher, 1e-5, np.inf)

            # Average over the examples in the batch
            reg_weights += [tf.reduce_mean(fisher, 0)]

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for v_g in v1_gs:
            fisher = v_g ** 2
            reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

        return reg_weights
