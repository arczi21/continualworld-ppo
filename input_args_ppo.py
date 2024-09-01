import argparse

from continualworld.tasks import TASK_SEQS
from continualworld.utils.enums import BufferType
from continualworld.utils.utils import float_or_str, sci2int, str2bool


def cl_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        type=str,
        choices=TASK_SEQS.keys(),
        default="CW5",
        help="Name of the sequence you want to run",
    )
    task_group.add_argument(
        "--task_list",
        nargs="+",
        default=None,
        help="List of tasks you want to run, by name or by the MetaWorld index",
    )
    parser.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        choices=["neptune", "tensorboard", "tsv"],
        default=["tsv", "neptune"],
        help="Types of logger used.",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="default_group",
        help="Group ID, for grouping logs from different experiments into common directory",
    )
    parser.add_argument("--seed", type=int, help="Seed for randomness")

    parser.add_argument(
        "--log_every",
        type=sci2int,
        default=int(5000),
        help="Number of steps between subsequent evaluations and logging",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes list for the MLP models",
    )
    parser.add_argument(
        "--activation", type=str, default="tanh", help="Activation kind for the models"
    )
    parser.add_argument(
        "--layer_norm",
        type=str2bool,
        default=True,
        help="Whether or not use layer normalization",
    )
    parser.add_argument("--pi_v_lr", type=float, default=1e-4, help="Learning rate for the policy and value optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_every", type=int, default=5000,
                        help="Number of environment interactions that should elapse between training epochs")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="Clipping in the policy objective")
    parser.add_argument("--train_pi_iters", type=int, default=256,
                        help="Maximum number of gradient descent steps to take on policy and value loss per epoch")
    parser.add_argument("--train_v_iters", type=int, default=256,
                        help="Maximum number of gradient descent steps to take on policy and value loss per epoch")
    parser.add_argument(
        "--steps_per_task", type=sci2int, default=int(1e6), help="Numer of steps per task"
    )
    parser.add_argument(
        "--reset_optimizer_on_task_change", #???
        type=str2bool,
        default=True,
        help="If true, optimizer is reset on each task change",
    )
    parser.add_argument(
        "--cl_method",
        type=str,
        choices=[None, "l2", "ewc", "mas", "vcl", "packnet", "agem", "episodic_replay"],
        default="packnet",
        help="If None, finetuning method will be used. If one of 'l2', 'ewc', 'mas', 'vcl',"
        "'packnet', 'agem', respective method will be used.",
    )
    parser.add_argument(
        "--regularize_critic",
        type=str2bool,
        default=False,
        help="If True, both actor and critic are regularized; if False, only actor is",
    )
    parser.add_argument(
        "--cl_reg_coef",
        type=float,
        default=1,
        help="Regularization strength for continual learning methods. "
        "Valid for 'l2', 'ewc', 'mas' continual learning methods.",
    )
    parser.add_argument(
        "--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture"
    )
    parser.add_argument(
        "--hide_task_id",
        type=str2bool,
        default=True,
        help="if True, one-hot encoding of the task will not be appended to observation",
    )
    parser.add_argument("--clipnorm", type=float, default=None, help="Value for gradient clipping")

    parser.add_argument(
        "--freeze_actor_on_task_change",
        type=str, default=None,
        choices=["core", "all"],
    )
    parser.add_argument(
        "--freeze_critic_on_task_change",
        type=str, default=None,
    )


    return parser.parse_args(args=args)


def mt_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        type=str,
        choices=TASK_SEQS.keys(),
        default=None,
        help="Name of the sequence you want to run",
    )
    task_group.add_argument(
        "--task_list",
        nargs="+",
        default=None,
        help="List of tasks you want to run, by name or by the MetaWorld index",
    )
    parser.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        choices=["neptune", "tensorboard", "tsv"],
        default=["tsv"],
        help="Types of logger used.",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="default_group",
        help="Group ID, for grouping logs from different experiments into common directory",
    )
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--steps_per_task", type=sci2int, default=int(1e6), help="Numer of steps per task"
    )
    parser.add_argument(
        "--log_every",
        type=sci2int,
        default=int(2e4),
        help="Number of steps between subsequent evaluations and logging",
    )
    parser.add_argument(
        "--replay_size", type=sci2int, default=int(1e6), help="Size of the replay buffer"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes list for the MLP models",
    )
    parser.add_argument(
        "--activation", type=str, default="tanh", help="Activation kind for the models"
    )
    parser.add_argument(
        "--layer_norm",
        type=str2bool,
        default=True,
        help="Whether or not use layer normalization",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--alpha",
        default="auto",
        help="Entropy regularization coefficient. "
        "Can be either float value, or 'auto', in which case it is dynamically tuned.",
    )
    parser.add_argument(
        "--target_output_std",
        type=float,
        default=0.089,
        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation "
        "of the action distribution on every dimension matches target_output_std.",
    )
    parser.add_argument(
        "--use_popart", type=str2bool, default=True, help="Whether use PopArt normalization"
    )
    parser.add_argument(
        "--popart_beta",
        type=float,
        default=3e-4,
        help="Beta parameter for updating statistics in PopArt",
    )
    parser.add_argument(
        "--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture"
    )
    parser.add_argument(
        "--hide_task_id",
        type=str2bool,
        default=True,
        help="if True, one-hot encoding of the task will not be appended to observation",
    )
    return parser.parse_args(args=args)


def single_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run single task")
    parser.add_argument("--task", type=str, help="Name of the task")
    parser.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        choices=["neptune", "tensorboard", "tsv"],
        default=["tsv"],
        help="Types of logger used.",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="default_group",
        help="Group ID, for grouping logs from different experiments into common directory",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")
    parser.add_argument(
        "--total_steps", type=sci2int, default=int(1e6), help="Number of steps the algorithm will run for"
    )
    parser.add_argument(
        "--log_every",
        type=sci2int,
        default=int(5000),
        help="Number of steps between subsequent evaluations and logging",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes list for the MLP models",
    )
    parser.add_argument(
        "--activation", type=str, default="tanh", help="Activation kind for the models"
    )
    parser.add_argument(
        "--layer_norm",
        type=str2bool,
        default=True,
        help="Whether or not use layer normalization",
    )
    parser.add_argument("--pi_v_lr", type=float, default=5e-4, help="Learning rate for the policy and value optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_every", type=int, default=5000, help="Number of environment interactions that should elapse between training epochs")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="Clipping in the policy objective")
    parser.add_argument("--train_pi_iters", type=int, default=80, help="Maximum number of gradient descent steps to take on policy and value loss per epoch")
    parser.add_argument("--train_v_iters", type=int, default=80, help="Maximum number of gradient descent steps to take on policy and value loss per epoch")

    return parser.parse_args(args=args)