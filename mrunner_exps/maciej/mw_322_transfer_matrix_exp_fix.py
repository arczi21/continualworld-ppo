from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults
from continualworld.tasks import TASK_SEQS

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "logger_output": ["tsv", "neptune"],
    "agent_policy_exploration": True,
    "cl_method": None,
}
config = combine_config_with_defaults(config)

tasks = TASK_SEQS["CW10"]
pairs = [[first_task, second_task] for first_task in tasks for second_task in tasks]
print(len(pairs), pairs)


params_grid = [
    {
        "seed": list(range(0, 20)),
        "task_list": pairs,
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

