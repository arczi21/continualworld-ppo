"""PPO algorithm implementation."""

import random
import time

import gym
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, Optional, Tuple, Union


from continualworld.ppo import core
from continualworld.utils_ppo import logx
from continualworld.utils.utils import reset_weights, set_seed




class PPOBuffer:
    """A buffer for storing trajectories experienced by a PPO agent.

    Uses Generalized Advantage Estimation (GAE-Lambda) for calculating
    the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Computes reward for unfinished trajectory.

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = \
            core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, reset=True):
        """Returns data stored in buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        if reset:
            assert self.ptr == self.max_size  # buffer has to be full
            self.ptr, self.path_start_idx = 0, 0

        # the next three lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return dict(
            observations=tf.convert_to_tensor(self.obs_buf),
            actions=tf.convert_to_tensor(self.act_buf),
            advantages=tf.convert_to_tensor(self.adv_buf),
            rtg=tf.convert_to_tensor(self.ret_buf),
            logp_old=tf.convert_to_tensor(self.logp_buf),
        )

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.max_size, size=batch_size)

        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.act_buf[idxs]),
            advantages=tf.convert_to_tensor(self.adv_buf[idxs]),
            returns=tf.convert_to_tensor(self.ret_buf[idxs]),
            logps=tf.convert_to_tensor(self.logp_buf[idxs]),
        )


class PPO:
    def __init__(self, env_fn, test_envs, total_steps, actor_cl: type = core.MlpActor, critic_cl: type = core.MlpCritic, ac_kwargs=None, seed=0,
        train_every=4000, log_every=4000, num_test_eps_stochastic=10, num_test_eps_deterministic=1, gamma=0.999,
        clip_ratio=0.2, pi_v_lr=3e-4, train_pi_iters=80, train_v_iters=80,
        lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=None, save_freq=int(1e4), save_path=None, clipnorm=None,
        reset_optimizer_on_task_change=False, reset_actor_on_task_change=False,
        reset_critic_on_task_change=False, freeze_actor_on_task_change=False, freeze_critic_on_task_change=False):

        self.env = env_fn
        self.test_envs = test_envs
        self.actor_cl = actor_cl
        self.critic_cl = critic_cl
        self.log_every = log_every
        self.num_test_eps_stochastic = num_test_eps_stochastic
        self.num_test_eps_deterministic = num_test_eps_deterministic
        self.clip_ratio = clip_ratio
        self.lam = lam
        self.target_kl = target_kl
        self.train_every = train_every
        self.max_ep_len = max_ep_len
        self.gamma = gamma
        self.clipnorm = clipnorm
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.total_steps = total_steps
        self.save_freq = save_freq
        self.save_path = save_path
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_actor_on_task_change = reset_actor_on_task_change
        self.reset_critic_on_task_change = reset_critic_on_task_change
        self.freeze_actor_on_task_change = freeze_actor_on_task_change
        self.freeze_critic_on_task_change = freeze_critic_on_task_change


        self.obs_dim = np.prod(self.env.observation_space.shape)
        self.act_dim = self.env.action_space.shape

        self.update_actor = True
        self.update_critic = True

        #config = locals()
        self.logger = logx.EpochLogger(**(logger_kwargs or {}))

        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.env.seed(seed)

        self.replay_buffer = PPOBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                              size=self.train_every, gamma=self.gamma, lam=self.lam)

        ac_kwargs = ac_kwargs or {}
        ac_kwargs["input_dim"] = self.obs_dim
        ac_kwargs['action_space'] = self.env.action_space

        self.actor = self.actor_cl(**ac_kwargs)
        del ac_kwargs["action_space"]

        self.critic = self.critic_cl(**ac_kwargs)

        self.actor_variables = self.actor.trainable_variables
        self.critic_variables = self.critic.trainable_variables

        self.all_common_variables = (self.actor.common_variables + self.critic.common_variables)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)
        self.pi_v_lr = pi_v_lr


    def adjust_gradients_pi(
        self,
        actor_gradients: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        return actor_gradients

    def adjust_gradients_v(
        self,
        critic_gradients: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        return critic_gradients

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        return tf.constant(0.0)

    def on_test_start(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_test_end(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_task_start(self, current_task_idx: int) -> None:
        pass

    def on_task_end(self, current_task_idx: int) -> None:
        pass

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        return None

    def get_action(self, observation, deterministic=False):
        return self.actor.action(np.array([observation]), deterministic).numpy()[0]

    def _log_after_pi_update(self, results):
        self.logger.store({"train/loss_pi": results["loss"].numpy(), "train/kl": results["kl"].numpy(),
                           "train/entropy": results["entropy"].numpy()})

    def _log_after_epoch(self, epoch, current_task_timestep, t, info):
        # Log info about epoch
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("total_env_steps", t + 1)
        self.logger.log_tabular("current_task_steps", current_task_timestep + 1)
        self.logger.log_tabular('train/return', with_min_and_max=True)
        self.logger.log_tabular('train/ep_length', average_only=True)
        self.logger.log_tabular('train/v_vals', with_min_and_max=True)
        self.logger.log_tabular('train/loss_v', average_only=True)
        self.logger.log_tabular('train/loss_pi', average_only=True)
        self.logger.log_tabular('train/entropy', average_only=True)
        self.logger.log_tabular('train/kl', average_only=True)
        self.logger.log_tabular('total_env_interacts', (t + 1))
        self.logger.log_tabular('walltime', time.time() - self.start_time)

        avg_success = np.mean(self.env.pop_successes())
        self.logger.log_tabular("train/success", avg_success)
        if "seq_idx" in info:
            self.logger.log_tabular("train/active_env", info["seq_idx"])

        self.logger.dump_tabular()

    def get_learn_pi(self, current_task_idx: int) -> Callable:
        @tf.function
        def learn_pi(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            gradients, metrics = self.get_gradients("pi", seq_idx, **batch)
            # Warning: we refer here to the int task_idx in the parent function, not
            # the passed seq_idx.
            gradients = self.adjust_gradients_pi(
                gradients,
                current_task_idx=current_task_idx,
                metrics=metrics,
                episodic_batch=episodic_batch,
            )

            if self.clipnorm is not None:
                actor_gradients = gradients
                gradients = tf.clip_by_global_norm(actor_gradients, self.clipnorm)[0]


            self.apply_update("pi", current_task_idx, gradients)
            return metrics

        return learn_pi

    def get_learn_v(self, current_task_idx: int) -> Callable:
        @tf.function
        def learn_v(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            gradients, loss = self.get_gradients("v", seq_idx, **batch)
            # Warning: we refer here to the int task_idx in the parent function, not
            # the passed seq_idx.
            gradients = self.adjust_gradients_v(
                gradients,
                current_task_idx=current_task_idx,
                metrics=None,
                episodic_batch=episodic_batch,
            )

            if self.clipnorm is not None:
                critic_gradients = gradients
                gradients = tf.clip_by_global_norm(critic_gradients, self.clipnorm)[0]

            self.apply_update("v", current_task_idx, gradients)
            return loss

        return learn_v

    def get_gradients(
        self,
        type: str,
        seq_idx: tf.Tensor,
        observations: tf.Tensor,
        actions: tf.Tensor,
        advantages: tf.Tensor,
        rtg: tf.Tensor,
        logp_old: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        if type == "pi":
            with tf.GradientTape(persistent=True) as g:
                # Main outputs from computation graph

                logp = self.actor.action_logprob(observations, actions)
                ratio = tf.exp(logp - logp_old)
                min_adv = tf.where(condition=(advantages >= 0),
                                   x=(1 + self.clip_ratio) * advantages,
                                   y=(1 - self.clip_ratio) * advantages)

                pi_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
                auxiliary_loss = self.get_auxiliary_loss(seq_idx)
                pi_loss += auxiliary_loss
            actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
            del g
            # For logging purposes
            logp = self.actor.action_logprob(observations, actions)
            ratio = tf.exp(logp - logp_old)
            min_adv = tf.where(condition=(advantages >= 0),
                               x=(1 + self.clip_ratio) * advantages,
                               y=(1 - self.clip_ratio) * advantages)
            loss_new = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
            metrics = dict(
                loss=loss_new,
                kl=tf.reduce_mean(logp_old - logp),
                entropy=tf.reduce_mean(-logp),
            )
            return actor_gradients, metrics

        elif type == "v":
            with tf.GradientTape(persistent=True) as g:
                # Main outputs from computation graph
                value_loss = tf.reduce_mean((self.critic(observations) - rtg) ** 2)
                aux_loss = self.get_auxiliary_loss(seq_idx)
                value_loss += aux_loss
            critic_gradients = g.gradient(value_loss, self.critic_variables)
            del g
            return critic_gradients, value_loss

    def apply_update(
        self,
        type: str,
        current_task_idx: int,
        gradients: List[tf.Tensor],
    ) -> None:

        if type == "pi":
            if self.update_actor:
                self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        if type == "v":
            if self.update_critic:
                self.critic_optimizer.apply_gradients(zip(gradients, self.critic_variables))



    def test_agent(self, deterministic, num_episodes) -> None:
        avg_success = []
        mode = "deterministic" if deterministic else "stochastic"
        for seq_idx, test_env in enumerate(self.test_envs):
            key_prefix = f"test/{mode}/{seq_idx}/{test_env.name}/"

            self.on_test_start(seq_idx)

            for j in range(num_episodes):
                obs, done, episode_return, episode_len = test_env.reset(), False, 0, 0
                while not (done or (episode_len == self.max_ep_len)):
                    obs, reward, done, _ = test_env.step(
                        self.get_action(tf.convert_to_tensor(obs), deterministic)
                    )
                    episode_return += reward
                    episode_len += 1
                self.logger.store(
                    {key_prefix + "return": episode_return, key_prefix + "ep_length": episode_len}
                )

            self.on_test_end(seq_idx)

            self.logger.log_tabular(key_prefix + "return", with_min_and_max=True)
            self.logger.log_tabular(key_prefix + "ep_length", average_only=True)
            env_success = test_env.pop_successes()
            avg_success += env_success
            self.logger.log_tabular(key_prefix + "success", np.mean(env_success))
        key = f"test/{mode}/average_success"
        self.logger.log_tabular(key, np.mean(avg_success))

    @tf.function
    def value(self, observations):
        return self.critic(observations)

    def get_value(self, observation):
        return self.value(np.array([observation])).numpy()[0]

    @tf.function
    def value_loss(self, observations, rtg):
        return tf.reduce_mean((self.critic(observations) - rtg) ** 2)

    @tf.function
    def value_train_step(self, observations, rtg, idx):
        def loss():
            aux_loss = self.get_auxiliary_loss(idx)
            return self.value_loss(observations, rtg) + aux_loss

        self.critic_optimizer.minimize(loss, self.critic.trainable_variables)

        return loss()

    @tf.function
    def pi_loss(self, logp, logp_old, advantages):
        ratio = tf.exp(logp - logp_old)
        min_adv = tf.where(condition=(advantages >= 0),
                           x=(1 + self.clip_ratio) * advantages,
                           y=(1 - self.clip_ratio) * advantages)
        return -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))

    @tf.function
    def pi_train_step(self, observations, actions, advantages, logp_old, idx):
        def loss():
            logp = self.actor.action_logprob(observations, actions)
            aux_loss = self.get_auxiliary_loss(idx)
            return self.pi_loss(logp, logp_old, advantages) + aux_loss

        self.actor_optimizer.minimize(loss, self.actor.trainable_variables)

        # For logging purposes
        logp = self.actor.action_logprob(observations, actions)
        loss_new = self.pi_loss(logp, logp_old, advantages)

        return loss_new, tf.reduce_mean(logp_old - logp), tf.reduce_mean(-logp)


    def _handle_task_change(self, current_task_idx: int):
        self.on_task_start(current_task_idx)

        if current_task_idx > 0:
            if self.reset_actor_on_task_change:
                if self.exploration_kind is not None:
                    self.exploration_actor.set_weights(self.actor.get_weights())
                reset_weights(self.actor, self.actor_cl, self.actor_kwargs)

            if self.reset_critic_on_task_change:
                reset_weights(self.critic, self.critic_cl, self.critic_kwargs)
                self.target_critic.set_weights(self.critic.get_weights())

            if self.reset_optimizer_on_task_change:
                self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_v_lr)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_v_lr)


            # Update variables list and update function in case model changed.
            # E.g: For VCL after the first task we set trainable=False for layer
            # normalization. We need to recompute the graph in order for TensorFlow
            # to notice this change.
            self.update_variables()


        self.learn_pi = self.get_learn_pi(current_task_idx)
        self.learn_v = self.get_learn_v(current_task_idx)


    def filter_variables(self, critic, layer_order, frozen_layers_num, total_layers_num=4):
        variables_to_train = []
        layer_idx = -1
        for variable in critic.core.trainable_variables:
            if "dense" in variable.name and "kernel" in variable.name:
                layer_idx += 1
            if layer_order == "f" and layer_idx >= frozen_layers_num:
                variables_to_train += [variable]
            elif layer_order == "l" and (total_layers_num - layer_idx) > frozen_layers_num:
                variables_to_train += [variable]


        return variables_to_train

    def update_variables(self):
        self.critic_variables = (
            self.critic.trainable_variables
            )
        if self.freeze_critic_on_task_change == "all":
            self.update_critic = False
        elif self.freeze_critic_on_task_change == "core":
            self.critic_variables = (
                self.critic.head.trainable_variables
                )
        elif (self.freeze_critic_on_task_change is not None and
                self.freeze_critic_on_task_change.startswith("core")):
            head_variables = (
                self.critic.head.trainable_variables
                )
            layer_order, layer_num = self.freeze_critic_on_task_change.split("-")[1]
            layer_num = int(layer_num)
            critic_trainable = self.filter_variables(self.critic, layer_order, layer_num)
            print("trainable critic1 variables", list(v.name for v in critic_trainable))
            self.critic_variables = self.critic_trainable + head_variables

        if self.freeze_critic_on_task_change is not None:
            self.target_critic.set_weights(self.critic.get_weights())


        self.actor_variables = self.actor.trainable_variables

        if self.freeze_actor_on_task_change == "all":
            self.update_actor = False
        elif self.freeze_actor_on_task_change == "core":
            self.actor_variables = (
                self.actor.head_mu.trainable_variables
                + self.actor._log_std.trainable_variables
            )

        self.all_common_variables = (
            self.actor.common_variables
            + self.critic.common_variables
        )


    def run(self):
        self.start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        current_task_timestep = 0
        current_task_idx = -1
        self.learn_pi = self.get_learn_pi(current_task_idx)
        self.learn_v = self.get_learn_v(current_task_idx)

        # Main loop: collect experience in env and update/log each epoch
        for t in range(self.total_steps):
            # On task change
            if current_task_idx != getattr(self.env, "cur_seq_idx", -1):
                current_task_timestep = 0
                current_task_idx = getattr(self.env, "cur_seq_idx")
                self._handle_task_change(current_task_idx)

            action = self.get_action(obs)
            v_t = self.get_value(obs)
            logp = self.actor.action_logprob(np.array([obs]),
                                        np.array([action])).numpy()[0]

            # Step the env
            new_obs, rew, done, info = self.env.step(action)
            ep_ret += rew
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if ep_len == self.max_ep_len else done

            # Store experience to replay buffer
            self.replay_buffer.store(obs, action, rew, v_t, logp)
            self.logger.store({"train/v_vals": v_t})

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = new_obs

            # End of trajectory handling
            if done or (ep_len == self.max_ep_len):
                self.logger.store({"train/return": ep_ret, "train/ep_length": ep_len})

            if done or (ep_len == self.max_ep_len) or (t + 1) % self.train_every == 0:
                obs, ep_ret, ep_len = self.env.reset(), 0, 0

                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.get_value(obs)
                self.replay_buffer.finish_path(last_val)


            # Update handling
            if (t + 1) % self.train_every == 0:
                batch = self.replay_buffer.get()
                episodic_batch = self.get_episodic_batch(current_task_idx)

                for i in range(self.train_pi_iters):
                    results = self.learn_pi(tf.convert_to_tensor(current_task_idx), batch, episodic_batch)
                    self._log_after_pi_update(results)
                    if results["kl"].numpy() > 1.5 * self.target_kl:
                        self.logger.log(
                            'Early stopping at step %d due to reaching max kl.' % i)
                        break

                for _ in range(self.train_v_iters):
                    loss = self.learn_v(tf.convert_to_tensor(current_task_idx), batch, episodic_batch)
                    self.logger.store({"train/loss_v": loss})

            if (
                self.env.name == "ContinualLearningEnv"
                and current_task_timestep + 1 == self.env.steps_per_env
            ):
                self.on_task_end(current_task_idx)

            # End of epoch wrap-up
            if ((t + 1) % self.log_every == 0) or (t + 1 == self.total_steps):
                epoch = (t + 1 + self.log_every - 1) // self.log_every

                # Test the performance of stochastic and deterministic version of the agent.
                self.test_agent(deterministic=False, num_episodes=self.num_test_eps_stochastic)
                self.test_agent(deterministic=True, num_episodes=self.num_test_eps_deterministic)

                self._log_after_epoch(epoch, current_task_timestep, t, info)


            # Save model
            if ((t + 1) % self.save_freq == 0) or (t + 1 == self.total_steps):
                if self.save_path is not None:
                    tf.keras.models.save_model(self.actor, self.save_path)

            current_task_timestep += 1
