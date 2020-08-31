import torch
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()


# noinspection PyUnusedLocal
@ex.config
def config():
    name = "run"
    run_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env_type = "1d"
    debug = False

    # Set up observer
    ex.observers.append(FileStorageObserver(f"./data/runs/{run_name}"))


# noinspection PyUnusedLocal
@ex.config
def env_config():
    illegal_reward=-50.0
    illegal_actions_patience=5

    n_max=10
    n_min=2
    n_target=1

    min_reward=-100.0
    state_rescaling=0.01
    padding_value=-1.0

    w_jet=True
    w_rate=3.0
    qcd_rate=1.5
    pt_min=4.0 ** 2
    qcd_mass=30.0
    w_mass=80.0
    jet_momentum=400.0
    jetdir=(1, 1, 1)
    max_n_try=1000


# noinspection PyUnusedLocal
@ex.config
def agent_config():
    reward_range = (-200., 0.)
    optim_kwargs = None
    history_length = None


# noinspection PyUnusedLocal
@ex.config
def train_config():
    train_n_mc_target = 5
    train_n_mc_min = 5
    train_n_mc_max = 100

    train_mcts_mode = "mean"
    train_c_puct = 1.0

    train_steps = 100000


# noinspection PyUnusedLocal
@ex.config
def eval_config():
    eval_n_mc_target = 10
    eval_n_mc_min = 10
    eval_n_mc_max = 250
    eval_mcts_mode = "mean"
    eval_c_puct = 1.0

    eval_jets = 8
    eval_repeats = 100

    eval_filename = "./data/eval/eval.pickle"
    eval_figure_path = "./figures/"

    redraw_eval_jets = False


# noinspection PyUnusedLocal
@ex.config
def technical_config():
    device = torch.device("cpu")
    dtype = torch.float
    seed = 1971248

