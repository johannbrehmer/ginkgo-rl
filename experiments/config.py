import torch
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

ex = Experiment(name="rl-ginkgo")
__all__ = ["ex", "config", "env_config", "agent_config", "train_config", "eval_config", "baseline_config", "technical_config"]


# noinspection PyUnusedLocal
@ex.config
def config():
    algorithm = "mcbs"  # one of "mcts", "mcbs", "greedy", "random"
    env_type = "1d"

    name = algorithm
    run_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set up observer
    ex.observers.append(FileStorageObserver(f"./data/runs/{run_name}"))
    ex.observers.append(MongoObserver())


# noinspection PyUnusedLocal
@ex.config
def env_config():
    illegal_reward=-100.0
    illegal_actions_patience=3

    n_max=20
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
    reward_range = (-500., 0.)
    history_length = None
    hidden_sizes = (100, 100,)
    activation = torch.nn.ReLU()


# noinspection PyUnusedLocal
@ex.config
def train_config():
    train_n_mc_target = 5
    train_n_mc_min = 1
    train_n_mc_max = 50
    train_beamsize = 10

    train_mcts_mode = "mean"
    train_c_puct = 1.0

    train_steps = 10000
    learning_rate = 1.0e-3
    lr_decay = 0.01
    weight_decay = 0.0


# noinspection PyUnusedLocal
@ex.config
def eval_config():
    eval_n_mc_target = 10
    eval_n_mc_min = 1
    eval_n_mc_max = 100
    eval_mcts_mode = "mean"
    eval_c_puct = 1.0
    eval_beamsize = 10

    eval_jets = 500
    eval_repeats = 1
    eval_filename = "./data/eval/eval.pickle"
    redraw_eval_jets = False


# noinspection PyUnusedLocal
@ex.config
def technical_config():
    device = torch.device("cpu")
    dtype = torch.float
    seed = 1971248
    debug = False


@ex.named_config
def debug():
    algorithm = "mcbs"
    env_type = "1d"
    name = "mcbs_debug"

    train_steps = 10
    train_beamsize = 2
    train_n_mc_min = 1
    train_n_mc_max = 2

    eval_beamsize = 2
    eval_n_mc_min = 1
    eval_n_mc_max = 2


@ex.named_config
def truth():
    algorithm = "truth"
    name = "truth"


@ex.named_config
def mle():
    algorithm = "mle"
    name = "mle"


@ex.named_config
def random():
    algorithm = "random"
    name = "random"


@ex.named_config
def greedy():
    algorithm = "greedy"
    name = "greedy"


@ex.named_config
def beamsearch_s():
    algorithm = "beamsearch"
    name = "beamsearch_s"
    eval_beamsize = 5


@ex.named_config
def beamsearch_m():
    algorithm = "beamsearch"
    name = "beamsearch_m"
    eval_beamsize = 20


@ex.named_config
def beamsearch_l():
    algorithm = "beamsearch"
    name = "beamsearch_l"
    eval_beamsize = 100


@ex.named_config
def mcbs_s():
    algorithm = "mcbs"
    name = "mcbs_s"

    train_beamsize = 5
    eval_beamsize = 5
    train_n_mc_target = 2
    eval_n_mc_target = 2
    train_n_mc_max = 40
    eval_n_mc_max = 40


@ex.named_config
def mcbs_m():
    algorithm = "mcbs"
    name = "mcbs_m"

    train_beamsize = 20
    eval_beamsize = 20
    train_n_mc_target = 5
    eval_n_mc_target = 5
    train_n_mc_max = 100
    eval_n_mc_max = 100


@ex.named_config
def mcbs_l():
    algorithm = "mcbs"
    name = "mcbs_l"

    train_beamsize = 100
    eval_beamsize = 100
    train_n_mc_target = 10
    eval_n_mc_target = 10
    train_n_mc_max = 200
    eval_n_mc_max = 200
