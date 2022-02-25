from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "logger_output": ["tsv", "neptune"],
    "episodic_memory_from_buffer": True,
    "tasks": "CW20",
    "agent_policy_exploration": True,
    "multihead_archs": True,
    "hide_task_id": True,
    "clipnorm": None,
}
config = combine_config_with_defaults(config)

params_grid = [
    {
        # Oracle ER
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "regularize_critic": [False],
        "episodic_mem_per_task": [10000],
        "cl_reg_coef": [100.],
        "episodic_batch_size": [128],
        "oracle_mode": [True],
        "oracle_sampling": [True],
        "oracle_clamp": [0.],
        "oracle_reuse_task": [True],
        "oracle_replay_new_head": [True, False],
    },
    {
        # Fine-tuning
        "seed": list(range(20)),
        "cl_method": [None],
        "regularize_critic": [False],
        "episodic_mem_per_task": [10000],
        "cl_reg_coef": [0.],
        "episodic_batch_size": [128],
        "oracle_mode": [False],
        "oracle_sampling": [False],
        "oracle_clamp": [0.],
        "oracle_reuse_task": [False],
        "oracle_replay_new_head": [False],
    },
    {
        # Regular ER
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "regularize_critic": [False],
        "episodic_mem_per_task": [10000],
        "cl_reg_coef": [100.],
        "episodic_batch_size": [128],
        "oracle_mode": [False],
        "oracle_sampling": [False],
        "oracle_clamp": [0.],
        "oracle_reuse_task": [False],
        "oracle_replay_new_head": [False],
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

