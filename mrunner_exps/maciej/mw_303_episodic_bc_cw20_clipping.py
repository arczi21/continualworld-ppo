from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "tasks": "CW20",
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = [
    {
        "seed": list(range(20)),
        "cl_method": ["episodic_replay"],
        "regularize_critic": [False, True],
        "episodic_mem_per_task": [100000],
        "cl_reg_coef": [1.],
        "episodic_batch_size": [128],
        "clipnorm": [2e-5, 1e-2, 1.]
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

