import torch
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
import logging

ex = Experiment(name="rl-ginkgo")
__all__ = ["ex", "config", "env_config", "agent_config", "train_config", "eval_config", "technical_config"]

logger = logging.getLogger(__name__)


# noinspection PyUnusedLocal
@ex.config
def config():
    algorithm = "mcts"  # {"mcts", "lfd", "lfd-mcts", "acer", "greedy", "random", "truth", "mle", "beamsearch"}
    policy = "nn"  # {"nn", "random", "likelihood"}
    teacher = "truth"  # {"truth", "mle"}

    env_type = "1d"  # for now, only {"1d"}
    seed = 24927  # 1971248 was used for first round

    name = f"{algorithm}_{policy}" if algorithm == "mcts" else algorithm
    run_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"

    database = True
    ex.observers.append(FileStorageObserver(f"./data/runs/{run_name}"))
    if database:
        ex.observers.append(MongoObserver())


# noinspection PyUnusedLocal
@ex.config
def env_config():
    illegal_reward = -100.0
    illegal_actions_patience = 3

    n_max = 20
    n_min = 2
    n_target = 1

    min_reward = -100.0
    state_rescaling = 0.01
    padding_value = -1.0

    w_jet = True
    w_rate = 3.0
    qcd_rate = 1.5
    pt_min = 4.0 ** 2
    qcd_mass = 30.0
    w_mass = 80.0
    jet_momentum = 400.0
    jetdir = (1, 1, 1)
    max_n_try = 1000


# noinspection PyUnusedLocal
@ex.config
def agent_config():
    initialize_mcts_with_beamsearch = True
    log_likelihood_policy_input = True
    decision_mode = "max_puct"  # {"max_reward", "max_puct", "mean_puct"}

    reward_range = (-500.0, 0.0)
    history_length = None
    hidden_sizes = (
        100,
        100,
    )
    activation = torch.nn.ReLU()


# noinspection PyUnusedLocal
@ex.config
def train_config():
    pretrain_steps = 50000
    pretrain_n_mc_target = 1
    pretrain_n_mc_min = 0
    pretrain_n_mc_max = 10
    pretrain_beamsize = 5
    pretrain_planning_mode = "mean"  # {"max", "mean"}, refers to PUCT
    pretrain_c_puct = 1.0

    train_steps = 10000
    train_n_mc_target = 1
    train_n_mc_min = 0
    train_n_mc_max = 20
    train_beamsize = 5
    train_planning_mode = "mean"  # {"max", "mean"}, refers to PUCT
    train_c_puct = 1.0

    imitation_steps = 500000

    learning_rate = 1.0e-3
    lr_decay = 0.01
    weight_decay = 0.0
    clip_gradient = None


# noinspection PyUnusedLocal
@ex.config
def eval_config():
    eval_n_mc_target = 5
    eval_n_mc_min = 0
    eval_n_mc_max = 100
    eval_beamsize = 10

    eval_planning_mode = "mean"
    eval_c_puct = 1.0

    eval_jets = 500
    eval_repeats = 1
    eval_filename = "./data/eval/eval2.pickle"
    redraw_eval_jets = False


# noinspection PyUnusedLocal
@ex.config
def technical_config():
    device = torch.device("cpu")
    dtype = torch.float
    debug = False
    debug_verbosity = 1


@ex.named_config
def debug_mcts():
    algorithm = "mcts"
    policy = "nn"
    name = "debug"
    debug = True

    eval_jets = 1
    eval_filename = "./data/eval/debug.pickle"
    redraw_eval_jets = True

    train_steps = 0
    train_beamsize = 3
    train_n_mc_min = 1
    train_n_mc_max = 2

    eval_beamsize = 3
    eval_n_mc_min = 1
    eval_n_mc_max = 2


@ex.named_config
def debug_lfd():
    algorithm = "lfd"
    name = "debug"
    debug = False

    imitation_steps = 10000


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
def beamsearch_xl():
    algorithm = "beamsearch"
    name = "beamsearch_xl"
    eval_beamsize = 1000


@ex.named_config
def mcts_xs():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_nn_xs"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_beamsize = 3
    train_n_mc_target = 1
    train_n_mc_max = 10

    eval_beamsize = 3
    eval_n_mc_target = 1
    eval_n_mc_max = 10


@ex.named_config
def mcts_s():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_nn_s"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_beamsize = 5
    train_n_mc_target = 1
    train_n_mc_max = 20

    eval_beamsize = 5
    eval_n_mc_target = 1
    eval_n_mc_max = 20


@ex.named_config
def mcts_m():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_nn_m"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_beamsize = 20
    train_n_mc_target = 2
    train_n_mc_max = 50

    eval_beamsize = 20
    eval_n_mc_target = 2
    eval_n_mc_max = 50


@ex.named_config
def mcts_l():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_nn_l"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_beamsize = 100
    train_n_mc_target = 5
    train_n_mc_max = 200

    eval_beamsize = 100
    eval_n_mc_target = 5
    eval_n_mc_max = 200


@ex.named_config
def lfd_s():
    algorithm = "lfd"
    name = "lfd_s"

    clip_gradient = 1.0


@ex.named_config
def lfd_mcts_s():
    algorithm = "lfd-mcts"
    name = "lfd-mcts_s"

    train_beamsize = 5
    train_n_mc_target = 1
    train_n_mc_max = 20

    eval_beamsize = 5
    eval_n_mc_target = 1
    eval_n_mc_max = 20

    clip_gradient = 1.0


@ex.named_config
def acer():
    algorithm = "acer"
    name = "acer"


@ex.named_config
def mcts_exploit():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_s_exploit"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10
    pretrain_c_puct = 0.1

    train_beamsize = 5
    train_n_mc_target = 1
    train_n_mc_max = 20
    train_c_puct = 0.1

    eval_beamsize = 5
    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_c_puct = 0.1


@ex.named_config
def mcts_explore():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_s_explore"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10
    pretrain_c_puct = 10.0

    train_beamsize = 5
    train_n_mc_target = 1
    train_n_mc_max = 20
    train_c_puct = 10.0

    eval_beamsize = 5
    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_c_puct = 10.0


@ex.named_config
def mcts_raw():
    algorithm = "mcts"
    policy = "nn"
    log_likelihood_policy_input = False
    name = "mcts_raw_s"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_n_mc_target = 1
    train_n_mc_max = 20
    train_beamsize = 5

    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_beamsize = 5


@ex.named_config
def mcts_puct_decisions():
    algorithm = "mcts"
    policy = "nn"
    name = "mcts_puct_decisions_s"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_n_mc_target = 1
    train_n_mc_max = 20
    train_beamsize = 5

    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_beamsize = 5

    decision_mode = "mean_puct"


@ex.named_config
def mcts_no_beamsearch():
    algorithm = "mcts"
    policy = "nn"
    initialize_mcts_with_beamsearch = False
    name = "mcts_nn_no_beamsearch_s"

    pretrain_beamsize = 3
    pretrain_n_mc_target = 1
    pretrain_n_mc_max = 10

    train_n_mc_target = 1
    train_n_mc_max = 20
    train_beamsize = 5

    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_beamsize = 5


@ex.named_config
def mcts_only_beamsearch():
    algorithm = "mcts"
    policy = "random"
    name = "mcts_only_bs_s"

    eval_n_mc_target = 0
    eval_n_mc_max = 0
    eval_beamsize = 5


@ex.named_config
def mcts_random():
    algorithm = "mcts"
    policy = "random"
    name = "mcts_random_s"

    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_beamsize = 5


@ex.named_config
def mcts_likelihood():
    algorithm = "mcts"
    policy = "likelihood"
    name = "mcts_likelihood_s"

    eval_n_mc_target = 1
    eval_n_mc_max = 20
    eval_beamsize = 5
