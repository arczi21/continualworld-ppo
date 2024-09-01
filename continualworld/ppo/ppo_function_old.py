"""PPO algorithm implementation."""

import random
import time

import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, Optional, Tuple, Union


from continualworld.ppo import core
from continualworld.utils_ppo import logx



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

    def get(self):
        """Returns data stored in buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        #adv_mean, adv_std = mpi_tools.mpi_statistics_scalar(self.adv_buf)

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf]


def ppo(env_fn, test_envs, total_steps, actor_cl: type = core.MlpActor, critic_cl: type = core.MlpCritic, ac_kwargs=None, seed=0,
        train_every=4000, log_every=4000, num_test_eps_stochastic=10, num_test_eps_deterministic=1, gamma=0.999,
        clip_ratio=0.2, pi_v_lr=3e-4, train_pi_iters=80, train_v_iters=80,
        lam=0.97, max_ep_len=1000, target_kl=0.01,
        logger_kwargs=None, save_freq=int(1e4), save_path=None,
        reset_optimizer_on_task_change=False):
    """Proximal Policy Optimization (by clipping).

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` and
            `observation_space` kwargs, and returns actor and critic
            tf.keras.Model-s.

            Actor implements method `action` which should take an observation
            in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ===========  ================  =====================================

            Furthermore, actor implements method `action_logprob` which should
            take an observation and an action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``logp``     (batch,)          | Gives the log probability of
                                           | selecting each specified action.
            ===========  ================  =====================================

            Critic should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``v``        (batch,)          | Gives estimate of current policy
                                           | value for states and actions in the
                                           | input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to PPO.

        seed (int): Seed for random number generators.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        train_every (int): Number of environment interactions that should elapse
            between training epochs.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        v_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        save_path (str): The path specifying where to save the trained actor
            model (note: path needs to point to a directory). Setting the value
            to None turns off the saving.
    """

    config = locals()
    logger = logx.EpochLogger(**(logger_kwargs or {}))

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


    env = env_fn
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape


    env.seed(seed)

    replay_buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim,
                              size=train_every, gamma=gamma, lam=lam)

    ac_kwargs = ac_kwargs or {}
    ac_kwargs["input_dim"] = obs_dim
    ac_kwargs['action_space'] = env.action_space

    actor = actor_cl(**ac_kwargs)
    del ac_kwargs["action_space"]

    critic = critic_cl(**ac_kwargs)

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)

    def get_action(observation, deterministic=False):
        return actor.action(np.array([observation]), deterministic).numpy()[0]

    def _log_after_epoch(epoch, current_task_timestep, t, info):
        # Log info about epoch
        logger.log_tabular("epoch", epoch)
        logger.log_tabular("total_env_steps", t + 1)
        logger.log_tabular("current_task_steps", current_task_timestep + 1)
        logger.log_tabular('train/return', with_min_and_max=True)
        logger.log_tabular('train/ep_length', average_only=True)
        logger.log_tabular('train/v_vals', with_min_and_max=True)
        logger.log_tabular('train/loss_v', average_only=True)
        logger.log_tabular('train/loss_pi', average_only=True)
        logger.log_tabular('train/entropy', average_only=True)
        logger.log_tabular('train/kl', average_only=True)
        logger.log_tabular('total_env_interacts', (t + 1))
        logger.log_tabular('walltime', time.time() - start_time)

        avg_success = np.mean(env.pop_successes())
        logger.log_tabular("train/success", avg_success)
        if "seq_idx" in info:
            logger.log_tabular("train/active_env", info["seq_idx"])

        logger.dump_tabular()


    def test_agent(deterministic, num_episodes) -> None:
        avg_success = []
        mode = "deterministic" if deterministic else "stochastic"
        for seq_idx, test_env in enumerate(test_envs):
            key_prefix = f"test/{mode}/{seq_idx}/{test_env.name}/"

            for j in range(num_episodes):
                obs, done, episode_return, episode_len = test_env.reset(), False, 0, 0
                while not (done or (episode_len == max_ep_len)):
                    obs, reward, done, _ = test_env.step(
                        get_action(tf.convert_to_tensor(obs), deterministic)
                    )
                    episode_return += reward
                    episode_len += 1
                logger.store(
                    {key_prefix + "return": episode_return, key_prefix + "ep_length": episode_len}
                )

            logger.log_tabular(key_prefix + "return", with_min_and_max=True)
            logger.log_tabular(key_prefix + "ep_length", average_only=True)
            env_success = test_env.pop_successes()
            avg_success += env_success
            logger.log_tabular(key_prefix + "success", np.mean(env_success))
        key = f"test/{mode}/average_success"
        logger.log_tabular(key, np.mean(avg_success))

    @tf.function
    def value(observations):
        return critic(observations)

    def get_value(observation):
        return value(np.array([observation])).numpy()[0]

    @tf.function
    def value_loss(observations, rtg):
        return tf.reduce_mean((critic(observations) - rtg) ** 2)

    @tf.function
    def value_train_step(observations, rtg):
        def loss():
            return value_loss(observations, rtg)

        critic_optimizer.minimize(loss, critic.trainable_variables)

        return loss()

    @tf.function
    def pi_loss(logp, logp_old, advantages):
        ratio = tf.exp(logp - logp_old)
        min_adv = tf.where(condition=(advantages >= 0),
                           x=(1 + clip_ratio) * advantages,
                           y=(1 - clip_ratio) * advantages)
        return -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))

    @tf.function
    def pi_train_step(observations, actions, advantages, logp_old):
        def loss():
            logp = actor.action_logprob(observations, actions)
            return pi_loss(logp, logp_old, advantages)

        actor_optimizer.minimize(loss, actor.trainable_variables)

        # For logging purposes
        logp = actor.action_logprob(observations, actions)
        loss_new = pi_loss(logp, logp_old, advantages)

        return loss_new, tf.reduce_mean(logp_old - logp), tf.reduce_mean(-logp)


    def _handle_task_change(self, current_task_idx: int):

        if current_task_idx > 0:
            if reset_actor_on_task_change:
                if exploration_kind is not None:
                    exploration_actor.set_weights(actor.get_weights())
                reset_weights(self.actor, actor_cl, actor_kwargs)

            if reset_critic_on_task_change:
                reset_weights(critic, critic_cl, critic_kwargs)
                self.target_critic.set_weights(self.critic.get_weights())


            if reset_optimizer_on_task_change:
                actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)
                critic_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)


            # Update variables list and update function in case model changed.
            # E.g: For VCL after the first task we set trainable=False for layer
            # normalization. We need to recompute the graph in order for TensorFlow
            # to notice this change.
            update_variables()


        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)

        if exploration_kind is not None and current_task_idx > 0:
            exploration_helper = ExplorationHelper(
                exploration_kind, num_available_heads=current_task_idx + 1
            )

    def update_variables(self):
        critic_variables = (
            critic.trainable_variables
            )
        if freeze_critic_on_task_change == "all":
            update_critic = False
        elif freeze_critic_on_task_change == "core":
            critic_variables = (
                critic.head.trainable_variables
                )
        elif (freeze_critic_on_task_change is not None and
                freeze_critic_on_task_change.startswith("core")):
            head_variables = (
                critic.head.trainable_variables
                )

            layer_order, layer_num = freeze_critic_on_task_change.split("-")[1]
            layer_num = int(layer_num)
            critic_trainable = filter_variables(critic, layer_order, layer_num)
            print("trainable critic1 variables", list(v.name for v in critic1_trainable))
            critic_variables = critic1_trainable + head_variables

        if freeze_critic_on_task_change is not None:
            target_critic.set_weights(critic.get_weights())


        actor_variables = actor.trainable_variables

        if freeze_actor_on_task_change == "all":
            update_actor = False
        elif freeze_actor_on_task_change == "core":
            actor_variables = (
                actor.head_mu.trainable_variables
                + actor.head_log_std.trainable_variables
            )

        all_common_variables = (
            actor.common_variables
            + self.critic1.common_variables
        )



    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    current_task_timestep = 0
    current_task_idx = -1

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # On task change
        if current_task_idx != getattr(env, "cur_seq_idx", -1):
            current_task_timestep = 0
            current_task_idx = getattr(env, "cur_seq_idx")
            #handle_task_change(current_task_idx)
            #handle_task_change
            if current_task_idx > 0:
                if reset_optimizer_on_task_change:
                    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)
                    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_v_lr)

        action = get_action(obs)
        v_t = get_value(obs)
        logp = actor.action_logprob(np.array([obs]),
                                    np.array([action])).numpy()[0]

        # Step the env
        new_obs, rew, done, info = env.step(action)
        ep_ret += rew
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len == max_ep_len else done

        # Store experience to replay buffer
        replay_buffer.store(obs, action, rew, v_t, logp)
        logger.store({"train/v_vals": v_t})

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = new_obs

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            logger.store({"train/return": ep_ret, "train/ep_length": ep_len})

        if done or (ep_len == max_ep_len) or (t + 1) % train_every == 0:
            obs, ep_ret, ep_len = env.reset(), 0, 0

            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = 0 if done else get_value(obs)
            replay_buffer.finish_path(last_val)


        # Update handling
        if (t + 1) % train_every == 0:
            [batch_obs, batch_act, batch_adv, batch_rtg,
             batch_logp] = replay_buffer.get()

            for i in range(train_pi_iters):
                loss, kl, entropy = pi_train_step(batch_obs, batch_act,
                                                  batch_adv, batch_logp)
                logger.store({"train/loss_pi": loss.numpy(), "train/kl": kl.numpy(),
                             "train/entropy": entropy.numpy()})

                if kl > 1.5 * target_kl:
                    logger.log(
                        'Early stopping at step %d due to reaching max kl.' % i)
                    break

            for _ in range(train_v_iters):
                loss = value_train_step(batch_obs, batch_rtg)
                logger.store({"train/loss_v": loss})

        # End of epoch wrap-up
        if ((t + 1) % log_every == 0) or (t + 1 == total_steps):
            epoch = (t + 1 + log_every - 1) // log_every

            # Test the performance of stochastic and detemi version of the agent.
            test_agent(deterministic=False, num_episodes=num_test_eps_stochastic)
            test_agent(deterministic=True, num_episodes=num_test_eps_deterministic)

            _log_after_epoch(epoch, current_task_timestep, t, info)


        # Save model
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if save_path is not None:
                tf.keras.models.save_model(actor, save_path)

        current_task_timestep += 1
