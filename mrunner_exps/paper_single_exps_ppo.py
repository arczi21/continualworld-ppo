from mrunner.helpers.specification_helper import create_experiments_helper

from continualworld.tasks import TASK_SEQS
from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl", #single
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = {
    "seed": list(range(20)),
    "tasks": ["CW5"], #task
    "pi_v_lr": [5e-5],
    "train_pi_iters": [80],
    "train_v_iters": [256],
    "regularize_critic": [False],
    "layer_norm": [False],
    "clipnorm": [2e-5],
    "retrain_pi_iters":[100],
    "retrain_v_iters": [100],
    #"cl_reg_coef": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5],
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="arczi21/continualworld",
    script="python3 mrunner_run_ppo.py",
    python_path=".",
    tags=[name, "v6", "ppo"],
    base_config=config,
    params_grid=params_grid,
)
