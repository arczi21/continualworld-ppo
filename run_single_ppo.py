from typing import Callable, Iterable

from continualworld.envs import get_single_env, get_task_name
#from continualworld.ppo.ppo import ppo
from continualworld.ppo.ppo import PPO
from continualworld.utils_ppo.logx import EpochLogger
from continualworld.utils.utils import get_activation_from_str
from input_args_ppo import single_parse_args


def main(
    logger_output,
    group_id,
    logger_kwargs,
    task: str,
    seed: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    layer_norm: bool,
    total_steps=int(1e6),
    train_every=5000,
    log_every=5000,
    num_test_eps_stochastic=10,
    num_test_eps_deterministic=1,
    gamma=0.99,
    clip_ratio=0.2,
    pi_v_lr=1e-4,
    train_pi_iters=256,
    train_v_iters=256,
    lam=0.95,
    max_ep_len=500,
    target_kl=0.01,
    save_freq=int(1e4),
    save_path=None
):
    ac_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        layer_norm=layer_norm,
    )

    #ppo(get_single_env(task), [get_single_env(task)], total_steps=total_steps, ac_kwargs=ac_kwargs, train_every=train_every, log_every=log_every, num_test_eps_stochastic=num_test_eps_stochastic, num_test_eps_deterministic=num_test_eps_deterministic, gamma=gamma,
    #    clip_ratio=clip_ratio, pi_v_lr=pi_v_lr, train_pi_iters=train_pi_iters,
    #    train_v_iters=train_v_iters, lam=lam, max_ep_len=max_ep_len, target_kl=target_kl,
    #    save_freq=save_freq, save_path=save_path, seed=seed, logger_kwargs=logger_kwargs)

    ppo = PPO(
        get_single_env(task), [get_single_env(task)], total_steps=total_steps, ac_kwargs=ac_kwargs,
        train_every=train_every, log_every=log_every, num_test_eps_stochastic=num_test_eps_stochastic,
        num_test_eps_deterministic=num_test_eps_deterministic, gamma=gamma,
        clip_ratio=clip_ratio, pi_v_lr=pi_v_lr, train_pi_iters=train_pi_iters,
        train_v_iters=train_v_iters, lam=lam, max_ep_len=max_ep_len, target_kl=target_kl,
        save_freq=save_freq, save_path=save_path, seed=seed, logger_kwargs=logger_kwargs
    )
    ppo.run()


if __name__ == "__main__":
    args = vars(single_parse_args())
    args["task"] = get_task_name(args["task"])
    logger_output = args["logger_output"]
    logger_output.append("neptune")
    group_id = args["group_id"]
    logger_kwargs = {
        "logger_output": logger_output,
        "config": args,
        "group_id": group_id
    }
    del args["group_id"]
    del args["logger_output"]
    main(logger_output = logger_output, group_id = group_id, logger_kwargs = logger_kwargs, **args)
