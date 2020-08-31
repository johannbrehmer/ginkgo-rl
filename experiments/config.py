import torch
from sacred import Experiment

ex = Experiment()

__all__ = ["ex", "config", "env_config", "mcts_config", "mcts_train_config", "technical_config"]


# noinspection PyUnusedLocal
@ex.config
def config():
    name = "model"
    env_type = "1d"
    debug = False


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
def mcts_config():
    reward_range = (-200., 0.)
    optim_kwargs = None
    history_length = None


# noinspection PyUnusedLocal
@ex.config
def mcts_train_config():
    n_mc_target = 5
    n_mc_min = 5
    n_mc_max = 100

    mcts_mode = "mean"
    c_puct = 1.0

    train_steps = 100000


# noinspection PyUnusedLocal
@ex.config
def technical_config():
    device = torch.device("cpu")
    dtype = torch.float
    seed = 1971248

