from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import gym
import metaworld
import numpy as np
from gym.wrappers import TimeLimit

from continualworld.utils.wrappers import OneHotAdder, RandomizationWrapper, SuccessCounter


def get_mt50() -> metaworld.MT50:
    saved_random_state = np.random.get_state()
    np.random.seed(1)
    MT50 = metaworld.MT50()
    np.random.set_state(saved_random_state)
    return MT50


MT50 = get_mt50()
META_WORLD_TIME_HORIZON = 200
MT50_TASK_NAMES = list(MT50.train_classes)
MW_OBS_LEN = 12
MW_ACT_LEN = 4

CW10_FT = np.array(
    [[0.4214,  0.1302,  0.6352,  0.5811,  0.2423,  0.4794,  0.0281, 0.5831,  0.3219,  0.4721],
     [0.128, -0.0508,  0.3174,  0.7584,  0.0099,  0.4623, -0.0369, 0.2845,  0.3573,  0.7814],
     [0.042, -0.0366,  0.5018,  0.7503,  0.0105,  0.2595, -0.1703, 0.319,  0.2658,  0.7867],
     [0.0313, -0.1492,  0.401,  0.4702,  0.0936,  0.4171,  0.0093, 0.3857,  0.3204,  0.3334],
     [0.1922,  0.2097,  0.4474,  0.4659,  0.4212,  0.4115,  0.1876, 0.5262,  0.3566,  0.3788],
     [0.0472, -0.0003,  0.5529,  0.6374,  0.0536,  0.511, -0.1405, 0.1982,  0.3208, -0.0101],
     [0.2034, -0.0569,  0.4706,  0.7824,  0.0869,  0.5516, -0.01, 0.2954,  0.4175,  0.8138],
     [0.2854, -0.0303,  0.3345,  0.1955,  0.144,  0.4179, -0.0554, 0.401,  0.33,  0.0969],
     [-0.0993,  0.0614,  0.2403,  0.689, -0.0116,  0.2189, -0.019, 0.4175,  0.3604,  0.8033],
     [0.1012, -0.0538,  0.2972,  0.5279, -0.0152,  0.1771, -0.2766, 0.2093,  0.312,  0.7374]]
)
CW10_FT_TRUNCATED = np.clip(CW10_FT, 0., np.inf)

CW20_FT = np.concatenate([CW10_FT, CW10_FT], 0)
CW20_FT = np.concatenate([CW20_FT, CW20_FT], 1)
CW20_FT_TRUNCATED = np.clip(CW20_FT, 0., np.inf)

CW20_REUSE_TASK_FIRST_HALF = np.concatenate([CW10_FT, CW10_FT], 0)
CW20_REUSE_TASK_SECOND_HALF = np.concatenate([np.identity(10), np.zeros([10, 10])], 0)
CW20_REUSE_TASK_FT = np.concatenate([CW20_REUSE_TASK_FIRST_HALF, CW20_REUSE_TASK_SECOND_HALF], 1)


TRIPLE_FT = np.array(
        [[0., 1., 1.],
         [0., 0., 0.],
         [0., 0., 0.]]
)


def get_task_name(name_or_number: Union[int, str]) -> str:
    try:
        index = int(name_or_number)
        return MT50_TASK_NAMES[index]
    except:
        return name_or_number


def set_simple_goal(env: gym.Env, name: str) -> None:
    goal = [task for task in MT50.train_tasks if task.env_name == name][0]
    env.set_task(goal)


def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]


def get_mt50_idx(env: gym.Env) -> int:
    idx = list(env._env_discrete_index.values())
    assert len(idx) == 1
    return idx[0]


def get_single_env(
    task: Union[int, str],
    one_hot_idx: int = 0,
    one_hot_len: int = 1,
    randomization: str = "random_init_all",
) -> gym.Env:
    """Returns a single task environment.

    Appends one-hot embedding to the observation, so that the model that operates on many envs
    can differentiate between them.

    Args:
      task: task name or MT50 number
      one_hot_idx: one-hot identifier (indicates order among different tasks that we consider)
      one_hot_len: length of the one-hot encoding, number of tasks that we consider
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: single-task environment
    """
    task_name = get_task_name(task)
    env = MT50.train_classes[task_name]()
    env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    env = OneHotAdder(env, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len)
    # Currently TimeLimit is needed since SuccessCounter looks at dones.
    env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    env = SuccessCounter(env)
    env.name = task_name
    env.num_envs = 1
    return env


def assert_equal_excluding_goal_dimensions(os1: gym.spaces.Box, os2: gym.spaces.Box) -> None:
    assert np.array_equal(os1.low[:9], os2.low[:9])
    assert np.array_equal(os1.high[:9], os2.high[:9])
    assert np.array_equal(os1.low[12:], os2.low[12:])
    assert np.array_equal(os1.high[12:], os2.high[12:])


def remove_goal_bounds(obs_space: gym.spaces.Box) -> None:
    obs_space.low[9:12] = -np.inf
    obs_space.high[9:12] = np.inf


class ContinualLearningEnv(gym.Env):
    def __init__(self, envs: List[gym.Env], steps_per_env: int) -> None:
        for i in range(len(envs)):
            assert envs[0].action_space == envs[i].action_space
            assert_equal_excluding_goal_dimensions(
                envs[0].observation_space, envs[i].observation_space
            )
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        remove_goal_bounds(self.observation_space)

        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def pop_successes(self) -> List[bool]:
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        obs, reward, done, info = self.envs[self.cur_seq_idx].step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to be shorter than 200.
            done = True
            info["TimeLimit.truncated"] = True

            self.cur_seq_idx += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self.envs[self.cur_seq_idx].reset()


def get_cl_env(
    tasks: List[Union[int, str]], steps_per_task: int, randomization: str = "random_init_all"
) -> gym.Env:
    """Returns continual learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: steps the agent will spend in each of single environments
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        env.name = task_name
        env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    cl_env = ContinualLearningEnv(envs, steps_per_task)
    cl_env.name = "ContinualLearningEnv"
    return cl_env


class MultiTaskEnv(gym.Env):
    def __init__(
        self, envs: List[gym.Env], steps_per_env: int, cycle_mode: str = "episode"
    ) -> None:
        assert cycle_mode == "episode"
        for i in range(len(envs)):
            assert envs[0].action_space == envs[i].action_space
            assert_equal_excluding_goal_dimensions(
                envs[0].observation_space, envs[i].observation_space
            )
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        remove_goal_bounds(self.observation_space)

        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.cycle_mode = cycle_mode

        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self._cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for MultiTaskEnv!")

    def pop_successes(self) -> List[bool]:
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        obs, reward, done, info = self.envs[self._cur_seq_idx].step(action)
        info["mt_seq_idx"] = self._cur_seq_idx
        if self.cycle_mode == "step":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        self.cur_step += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        if self.cycle_mode == "episode":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        obs = self.envs[self._cur_seq_idx].reset()
        return obs


def get_mt_env(
    tasks: List[Union[int, str]], steps_per_task: int, randomization: str = "random_init_all"
):
    """Returns multi-task learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: agent will be limited to steps_per_task * len(tasks) steps
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        env.name = task_name
        env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    mt_env = MultiTaskEnv(envs, steps_per_task)
    mt_env.name = "MultiTaskEnv"
    return mt_env
