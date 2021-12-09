from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from continualworld.sac.replay_buffers import EpisodicMemory
from continualworld.sac.sac import SAC


class Episodic_SAC(SAC):
    def __init__(
        self, episodic_mem_per_task: int = 0, episodic_batch_size: int = 0, **vanilla_sac_kwargs
    ):
        """Episodic replay.

        Args:
          episodic_mem_per_task: Number of examples to keep in additional memory per task.
          episodic_batch_size: Minibatch size to compute additional loss.
        """
        super().__init__(**vanilla_sac_kwargs)
        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size

        episodic_mem_size = self.episodic_mem_per_task * self.env.num_envs
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
        )

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        if current_task_idx > 0:
            (ref_actor_gradients, ref_critic_gradients, _), _ = self.get_gradients(
                seq_idx=tf.constant(-1), **episodic_batch
            )

            final_actor_gradients = self.merge_gradients(actor_gradients, ref_actor_gradients)
            final_critic_gradients = self.merge_gradients(critic_gradients, ref_critic_gradients)
        else:
            final_actor_gradients = actor_gradients
            final_critic_gradients = critic_gradients

        return final_actor_gradients, final_critic_gradients, alpha_gradient


    def merge_gradients(self, new_grads: List[tf.Tensor], ref_grads: List[tf.Tensor]):
        final_grads = []
        for new_grad, ref_grad in zip(new_grads, ref_grads):
          final_grads += [(new_grad + ref_grad) / 2]
        return final_grads

    def on_task_start(self, current_task_idx: int) -> None:
        if current_task_idx > 0:
            new_episodic_mem = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
            self.episodic_memory.store_multiple(**new_episodic_mem)

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0:
            return self.episodic_memory.sample_batch(self.episodic_batch_size)
        return None
