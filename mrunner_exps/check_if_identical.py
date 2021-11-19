from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "tasks": "CW20",
    "logger_output": ["tsv", "neptune"],
    "steps_per_task": 40_000,
    "seed": 0,
}
config = combine_config_with_defaults(config)

params_grid = [
    {  # fine-tuning
        "cl_method": [None],
    },
    {
        "cl_method": ["l2"],
        "cl_reg_coef": [1e5],
    },
    {
        "cl_method": ["ewc"],
        "cl_reg_coef": [1e4],
    },
    {
        "cl_method": ["mas"],
        "cl_reg_coef": [1e4],
    },
    {
        "cl_method": ["vcl"],
        "cl_reg_coef": [1.0],
        "vcl_first_task_kl": [False],
    },
    {
        "cl_method": ["packnet"],
        "packnet_retrain_steps": [1000],
        "clipnorm": [2e-5],
    },
    {  # perfect memory
        "cl_method": [None],
        "batch_size": [512],
        "buffer_type": ["reservoir"],
        "reset_buffer_on_task_change": [False],
        "replay_size": [20_000_000],
    },
    {
        "cl_method": ["agem"],
        "regularize_critic": [True],
        "episodic_mem_per_task": [10000],
        "episodic_batch_size": [128],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="pmtest/continual-learning",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name, "v6"],
    base_config=config,
    params_grid=params_grid,
)
