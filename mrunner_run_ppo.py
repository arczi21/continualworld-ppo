from mrunner.helpers.client_helper import get_configuration
import mrunner
mrunner.settings.NEPTUNE_USE_NEW_API = True

from continualworld.envs import get_task_name
from run_cl_ppo import main as cl_main
from run_mt import main as mt_main
from run_single_ppo import main as single_main


MAIN_DICT = {"cl": cl_main, "mt": mt_main, "single": single_main}

if __name__ == "__main__":
    config = get_configuration(print_diagnostics=True, with_neptune=True)


    del config["experiment_id"]
    run_kind = config.pop("run_kind")

    if run_kind == "single":
        config["task"] = get_task_name(config["task"])

    logger_output = config.pop("logger_output")
    group_id = config.pop("group_id")

    logger_kwargs = {
        "logger_output": logger_output,
        "config": config,
        "group_id": group_id,
        "with_mrunner": True
    }

    main = MAIN_DICT[run_kind]
    main(logger_output=logger_output, group_id=group_id, logger_kwargs=logger_kwargs, **config)
