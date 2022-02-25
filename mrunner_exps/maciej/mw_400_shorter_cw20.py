from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "tasks": "CW20",
    "steps_per_task": int(5e5),
    "logger_output": ["tsv", "neptune"],
    "oracle_mode": False,
    "oracle_reuse_task": True,
}
config = combine_config_with_defaults(config)

params_grid = [
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "episodic_mem_per_task": [10000],
        "cl_reg_coef": [100],
        "episodic_batch_size": [128],
        "regularize_critic": [False],
        "clipnorm": [None, 0.01],
        "agent_policy_exploration": [True, False],
    },
    {  # fine-tuning
        "seed": list(range(20)),
        "cl_method": [None],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["l2"],
        "cl_reg_coef": [1e5],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["ewc"],
        "cl_reg_coef": [1e4],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["mas"],
        "cl_reg_coef": [1e4],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["vcl"],
        "cl_reg_coef": [1.0],
        "vcl_first_task_kl": [False],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["packnet"],
        "packnet_retrain_steps": [100000],
        "clipnorm": [2e-5],
        "agent_policy_exploration": [True, False],
    },
    {  # perfect memory
        "seed": list(range(20)),
        "cl_method": [None],
        "batch_size": [512],
        "buffer_type": ["reservoir"],
        "reset_buffer_on_task_change": [False],
        "replay_size": [20_000_000],
        "agent_policy_exploration": [True, False],
    },
    {
        "seed": list(range(20)),
        "cl_method": ["agem"],
        "regularize_critic": [True],
        "episodic_mem_per_task": [10000],
        "episodic_batch_size": [128],
        "agent_policy_exploration": [True, False],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="pmtest/continual-learning-2",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name, "v7"],
    base_config=config,
    params_grid=params_grid,
)
