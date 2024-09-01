from typing import Callable, Iterable, List

from continualworld.ppo.ppo_function_old import ppo
from continualworld.envs import get_cl_env, get_single_env
from continualworld.methods.vcl import VclMlpActor
from continualworld.ppo.core import MlpActor
from continualworld.utils_ppo.logx import EpochLogger
from continualworld.tasks import TASK_SEQS
from continualworld.utils.enums import BufferType
from continualworld.utils.run_utils import get_ppo_class
from continualworld.utils.utils import get_activation_from_str
from input_args_ppo import cl_parse_args


def main(
    logger_output,
    group_id,
    logger_kwargs,
    tasks: str,
    task_list: List[str],
    seed: int,
    hidden_sizes: Iterable[int],
    reset_optimizer_on_task_change: bool,
    activation: Callable,
    layer_norm: bool,
    multihead_archs: bool,
    hide_task_id: bool,
    freeze_actor_on_task_change: str,
    freeze_critic_on_task_change: str,
    reset_actor_on_task_change = False,
    reset_critic_on_task_change = False,
    regularize_critic = False,
    retrain_pi_iters=1e4,
    retrain_v_iters=0,
    cl_reg_coef = 1,
    cl_method = 3,
    steps_per_task=int(1e6),
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
    save_path=None,
    clipnorm=None
):

    assert (tasks is None) != (task_list is None)
    if tasks is not None:
        tasks = TASK_SEQS[tasks]
    else:
        tasks = task_list
    train_env = get_cl_env(tasks, steps_per_task)
    # Consider normalizing test envs in the future.
    num_tasks = len(tasks)
    test_envs = [
        get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks)
    ]
    total_steps = steps_per_task * len(tasks)

    if cl_method is not None:
        no_freezing = (freeze_actor_on_task_change is None and freeze_critic_on_task_change is None)
        assert no_freezing, "CL methods with freezing are not supported yet"

    num_heads = num_tasks if multihead_archs else 1
    ac_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        layer_norm=layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )
    if cl_method == "vcl":
        actor_cl = VclMlpActor
    else:
        actor_cl = MlpActor

    vanilla_ppo_kwargs = {
        "env_fn": train_env,
        "test_envs": test_envs,
        "total_steps": total_steps,
        "actor_cl": actor_cl,
        "train_every": train_every,
        "seed": seed,
        "log_every": log_every,
        "num_test_eps_stochastic": num_test_eps_stochastic,
        "num_test_eps_deterministic": num_test_eps_deterministic,
        "ac_kwargs": ac_kwargs,
        "reset_optimizer_on_task_change": reset_optimizer_on_task_change,
        "pi_v_lr": pi_v_lr,
        "reset_actor_on_task_change": reset_actor_on_task_change,#???
        "reset_critic_on_task_change": reset_critic_on_task_change,#???
        "clipnorm": clipnorm, #to w learn_on_batch
        "clip_ratio": clip_ratio,
        "lam": lam,
        "max_ep_len": max_ep_len,
        "target_kl": target_kl,
        "gamma": gamma,
        "train_pi_iters": train_pi_iters,
        "train_v_iters": train_v_iters,
        "freeze_actor_on_task_change": freeze_actor_on_task_change,
        "freeze_critic_on_task_change": freeze_critic_on_task_change,
        "save_freq": save_freq,
        "save_path": save_path,
        "logger_kwargs": logger_kwargs,
    }

    ppo_class = get_ppo_class(cl_method)

    if cl_method is None:
        ppo = ppo_class(**vanilla_ppo_kwargs)

    elif cl_method in ["l2", "ewc", "mas"]:
        ppo = ppo_class(
            **vanilla_ppo_kwargs, cl_reg_coef=cl_reg_coef, regularize_critic=regularize_critic
        )
    elif cl_method == "vcl":
        ppo = ppo_class(
            **vanilla_ppo_kwargs,
            cl_reg_coef=cl_reg_coef,
            regularize_critic=regularize_critic,
            first_task_kl=vcl_first_task_kl,
        )
    elif cl_method == "packnet":
        ppo = ppo_class(
            **vanilla_ppo_kwargs,
            regularize_critic=regularize_critic,
            retrain_pi_iters=retrain_pi_iters,
            retrain_v_iters=retrain_v_iters,
        )
    elif cl_method == "agem":
        ppo = ppo_class(
            **vanilla_ppo_kwargs,
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
        )
    elif cl_method == "episodic_replay":
        ppo = ppo_class(
            **vanilla_ppo_kwargs,
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
            episodic_memory_from_buffer=episodic_memory_from_buffer,
            regularize_critic=regularize_critic,
            cl_reg_coef=cl_reg_coef,
        )
    else:
        raise NotImplementedError("This method is not implemented")
    ppo.run()

if __name__ == "__main__":
    args = vars(cl_parse_args())
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
    main(logger_output=logger_output, group_id=group_id, logger_kwargs=logger_kwargs, **args)
