from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "logger_output": ["tsv", "neptune"],
    "episodic_memory_from_buffer": True,
    "tasks": "CW20",
    "agent_policy_exploration": True,
}
config = combine_config_with_defaults(config)

params_grid = [
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "regularize_critic": [False],
        "episodic_mem_per_task": [10000],
        "cl_reg_coef": [100],
        "episodic_batch_size": [128],
        "clipnorm": [None],
        "oracle_mode": [False],
        "oracle_reuse_task": [True, False],
        "start_steps_second_half": [int(1e3), int(1e4), int(1e5), int(1e6)],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="pmtest/continual-learning-2",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name, "v6"],
    base_config=config,
    params_grid=params_grid,
)

